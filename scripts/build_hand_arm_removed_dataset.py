#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, os
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)

INFER_W = 320
INFER_H = 568
BATCH_SIZE = 12
PERSON_PROB_THRESHOLD = 0.25
MIN_COMPONENT_PIXELS = 18
DILATE_KERNEL = 7
INPAINT_RADIUS = 5

def resolve(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    return root / "images" / rel

def feat(arr_bgr: np.ndarray):
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness_laplacian_var": float(lap.var()),
        "entropy_bits": entropy,
        "edge_density": float((edges > 0).mean()),
    }

def summarize(vals):
    a=np.asarray(vals,dtype=float)
    return {
        "n":int(len(a)), "mean":float(a.mean()), "std":float(a.std()),
        "min":float(a.min()), "q1":float(np.quantile(a,.25)),
        "median":float(np.median(a)), "q3":float(np.quantile(a,.75)),
        "max":float(a.max())
    }

def clean_lowres_mask(mask: np.ndarray) -> np.ndarray:
    m=(mask.astype(np.uint8)*255)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
    n,lab,stats,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    out=np.zeros_like(m)
    for k in range(1,n):
        if int(stats[k,cv2.CC_STAT_AREA]) >= MIN_COMPONENT_PIXELS:
            out[lab==k]=255
    out=cv2.dilate(out,np.ones((DILATE_KERNEL,DILATE_KERNEL),np.uint8),iterations=1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--clean-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args()
    args.clean_root.mkdir(parents=True,exist_ok=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)

    with args.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))

    weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    categories=list(weights.meta["categories"])
    person_idx=categories.index("person")

    model=lraspp_mobilenet_v3_large(weights=weights).eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))

    mean=torch.tensor([0.485,0.456,0.406],dtype=torch.float32).view(3,1,1)
    std=torch.tensor([0.229,0.224,0.225],dtype=torch.float32).view(3,1,1)

    manifests=[]
    features=[]
    qa=[]

    for start in range(0,len(rows),BATCH_SIZE):
        batch=rows[start:start+BATCH_SIZE]
        tensors=[]
        originals=[]
        srcs=[]

        for r in batch:
            src=resolve(args.image_root,r["path"])
            srcs.append(src)
            with Image.open(src) as im:
                rgb=im.convert("RGB")
                originals.append(np.asarray(rgb,dtype=np.uint8))
                small=rgb.resize((INFER_W,INFER_H),Image.Resampling.BILINEAR)
            arr=np.asarray(small,dtype=np.float32)/255.0
            ten=torch.from_numpy(arr).permute(2,0,1)
            tensors.append((ten-mean)/std)

        x=torch.stack(tensors,dim=0)
        logits=model(x)["out"]
        person_prob=torch.softmax(logits,dim=1)[:,person_idx].cpu().numpy()

        for bi,(r,src,rgb) in enumerate(zip(batch,srcs,originals)):
            H,W=rgb.shape[:2]
            low=clean_lowres_mask(person_prob[bi] >= PERSON_PROB_THRESHOLD)
            mask=cv2.resize(low,(W,H),interpolation=cv2.INTER_NEAREST)

            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            if np.any(mask):
                cleaned=cv2.inpaint(bgr,mask,INPAINT_RADIUS,cv2.INPAINT_TELEA)
            else:
                cleaned=bgr.copy()

            dst=args.clean_root/r["path"]
            dst.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dst),cleaned,[int(cv2.IMWRITE_JPEG_QUALITY),95])

            cov=float((mask>0).mean())
            fo=feat(bgr); fc=feat(cleaned)
            rec={
                "path":r["path"],"label":r["label"],"split":r["split"],
                "video_id":r["video_id"],"frame_num":r["frame_num"],
                "width":W,"height":H,
                "mask_fraction":cov,"changed":int(cov>0),
                "original_bytes":int(src.stat().st_size),
                "cleaned_bytes":int(dst.stat().st_size),
            }
            manifests.append(rec)
            features.append({
                **rec,
                **{f"orig_{k}":v for k,v in fo.items()},
                **{f"clean_{k}":v for k,v in fc.items()},
                **{f"delta_{k}":fc[k]-fo[k] for k in fo}
            })
            if cov>0:
                qa.append((cov,r["path"],str(src),str(dst)))
        print(f"processed {min(start+BATCH_SIZE,len(rows))}/{len(rows)}",flush=True)

    def write_csv(path,data):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    write_csv(args.out_dir/"cleaned_manifest.csv",manifests)
    write_csv(args.out_dir/"visual_features_original_vs_cleaned.csv",features)

    keys=["brightness_mean","contrast_std","sharpness_laplacian_var","entropy_bits","edge_density"]
    summary={
        "image_count":len(rows),
        "method":"Torchvision LRASPP MobileNetV3-Large semantic person segmentation + mask cleanup/dilation + OpenCV TELEA inpainting",
        "person_probability_threshold":PERSON_PROB_THRESHOLD,
        "inference_size":[INFER_W,INFER_H],
        "changed_image_count":sum(x["changed"] for x in manifests),
        "unchanged_image_count":sum(1-x["changed"] for x in manifests),
        "mask_fraction":summarize([x["mask_fraction"] for x in manifests]),
        "by_label":{
            lab:{
              "n":sum(x["label"]==lab for x in manifests),
              "changed":sum(x["label"]==lab and x["changed"] for x in manifests),
              "mask_fraction":summarize([x["mask_fraction"] for x in manifests if x["label"]==lab])
            } for lab in ["flip","notflip"]
        },
        "dimensions":{
            "unique_widths_original":sorted(set(int(x["width"]) for x in manifests)),
            "unique_heights_original":sorted(set(int(x["height"]) for x in manifests)),
            "cleaned_preserves_dimensions":all(
                cv2.imread(str(args.clean_root/x["path"])).shape[:2]==(int(x["height"]),int(x["width"]))
                for x in manifests
            )
        },
        "jpeg_bytes":{
            "original":summarize([x["original_bytes"] for x in manifests]),
            "cleaned":summarize([x["cleaned_bytes"] for x in manifests]),
            "paired_delta":summarize([x["cleaned_bytes"]-x["original_bytes"] for x in manifests])
        },
        "features":{}
    }
    for k in keys:
        summary["features"][k]={
            "original":summarize([x[f"orig_{k}"] for x in features]),
            "cleaned":summarize([x[f"clean_{k}"] for x in features]),
            "paired_delta_clean_minus_original":summarize([x[f"delta_{k}"] for x in features]),
            "original_flip":summarize([x[f"orig_{k}"] for x in features if x["label"]=="flip"]),
            "original_notflip":summarize([x[f"orig_{k}"] for x in features if x["label"]=="notflip"]),
            "cleaned_flip":summarize([x[f"clean_{k}"] for x in features if x["label"]=="flip"]),
            "cleaned_notflip":summarize([x[f"clean_{k}"] for x in features if x["label"]=="notflip"]),
        }

    summary["scope_note"]="Derived ablation dataset only. Canonical images are unchanged. Inpainting may introduce artifacts, so this dataset is used for sensitivity testing, not as a new ground truth."
    (args.out_dir/"hand_arm_cleaning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # QA sheet: 12 largest masks + 12 around median nonzero.
    qa=sorted(qa,reverse=True,key=lambda x:x[0])
    picks=qa[:12]
    if len(qa)>24:
        mid=len(qa)//2
        picks += qa[max(0,mid-6):mid+6]
    if picks:
        cell=(180,320)
        examples_per_row=2
        rowsheet=math.ceil(len(picks)/examples_per_row)
        sheet=Image.new("RGB",(examples_per_row*2*cell[0],rowsheet*cell[1]),"white")
        for k,(_,rel,src,dst) in enumerate(picks):
            rr=k//examples_per_row; ex=k%examples_per_row
            for offset,p in [(0,src),(1,dst)]:
                with Image.open(p) as im:
                    t=ImageOps.fit(im.convert("RGB"),cell,method=Image.Resampling.BILINEAR)
                sheet.paste(t,((ex*2+offset)*cell[0],rr*cell[1]))
        sheet.save(args.out_dir/"hand_arm_cleaning_qa.jpg",quality=90)

    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
