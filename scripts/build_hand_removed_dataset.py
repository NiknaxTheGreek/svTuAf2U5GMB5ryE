#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, os
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def skin_mask_bordered(img_bgr: np.ndarray) -> np.ndarray:
    h,w=img_bgr.shape[:2]
    small=cv2.resize(img_bgr,(max(1,w//2),max(1,h//2)),interpolation=cv2.INTER_AREA)
    sh,sw=small.shape[:2]

    ycrcb=cv2.cvtColor(small,cv2.COLOR_BGR2YCrCb)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    Y,Cr,Cb=cv2.split(ycrcb)
    H,S,V=cv2.split(hsv)

    # Broad skin candidate ensemble, tuned to favor recall.
    ycc=(Y>35)&(Cr>=130)&(Cr<=185)&(Cb>=70)&(Cb<=140)
    hsv_skin=((H<=28)|(H>=170))&(S>=18)&(S<=210)&(V>=45)
    cand=(ycc & hsv_skin).astype(np.uint8)*255

    k1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,k1)
    cand=cv2.morphologyEx(cand,cv2.MORPH_CLOSE,k2)

    # Keep only sizeable skin components plausibly entering from an image boundary.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand,8)
    keep=np.zeros_like(cand)
    margin=max(12,int(0.035*min(sh,sw)))
    min_area=max(180,int(0.0015*sh*sw))
    for k in range(1,n):
        x,y,ww,hh,area=stats[k]
        touches=(x<=margin or y<=margin or x+ww>=sw-margin or y+hh>=sh-margin)
        lower_or_side=(y+hh>=int(0.55*sh) or x<=margin or x+ww>=sw-margin)
        if area>=min_area and touches and lower_or_side:
            keep[labels==k]=255

    # Expand around fingers/forearms and bridge small holes.
    kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    keep=cv2.dilate(keep,kd,iterations=1)
    keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,kd)

    full=cv2.resize(keep,(w,h),interpolation=cv2.INTER_NEAREST)
    return full

def inpaint_person_region(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h,w=img_bgr.shape[:2]
    # Work at half resolution for speed, then composite only the edited region.
    small=cv2.resize(img_bgr,(w//2,h//2),interpolation=cv2.INTER_AREA)
    smask=cv2.resize(mask,(w//2,h//2),interpolation=cv2.INTER_NEAREST)
    filled=cv2.inpaint(small,smask,5,cv2.INPAINT_TELEA)
    filled_full=cv2.resize(filled,(w,h),interpolation=cv2.INTER_CUBIC)
    m=(mask.astype(np.float32)/255.0)[...,None]
    # feather edge to avoid a sharp binary seam
    feather=cv2.GaussianBlur(mask,(0,0),sigmaX=4).astype(np.float32)/255.0
    feather=feather[...,None]
    out=(img_bgr.astype(np.float32)*(1-feather)+filled_full.astype(np.float32)*feather)
    return np.clip(out,0,255).astype(np.uint8)

def features(img_bgr: np.ndarray):
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    brightness=float(gray.mean())
    contrast=float(gray.std())
    sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    sat=float(hsv[:,:,1].mean())
    edge=float((cv2.Canny(gray,80,160)>0).mean())
    return dict(
        brightness_mean=brightness,
        contrast_std=contrast,
        sharpness_laplacian_var=sharp,
        saturation_mean=sat,
        edge_density=edge,
    )

def make_sheet(items, out):
    if not items: return
    thumb=(180,320); gap=10; label_h=34; cols=4
    rows=math.ceil(len(items)*2/cols)
    canvas=Image.new("RGB",(cols*thumb[0]+(cols+1)*gap,rows*(thumb[1]+label_h)+(rows+1)*gap),"white")
    draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    cell=0
    for name,before,after,cov in items:
        for tag,arr in [("before",before),("cleaned",after)]:
            r,c=divmod(cell,cols)
            x=gap+c*(thumb[0]+gap); y=gap+r*(thumb[1]+label_h+gap)
            im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
            im=ImageOps.fit(im,thumb,method=Image.Resampling.BILINEAR)
            canvas.paste(im,(x,y))
            draw.text((x,y+thumb[1]+4),f"{name} {tag} mask={cov:.1%}",fill="black",font=font)
            cell+=1
    canvas.save(out,quality=90)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-root",required=True,type=Path)
    ap.add_argument("--report-dir",required=True,type=Path)
    a=ap.parse_args()
    a.out_root.mkdir(parents=True,exist_ok=True)
    a.report_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))

    manifest=[]
    feats=[]
    preview=[]
    detected=0
    for idx,r in enumerate(rows):
        src=a.image_root/r["path"]
        if not src.exists(): src=a.image_root/"images"/r["path"]
        img=cv2.imread(str(src),cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError(f"Could not read {src}")
        mask=skin_mask_bordered(img)
        cov=float((mask>0).mean())
        if cov>0:
            detected+=1
            clean=inpaint_person_region(img,mask)
        else:
            clean=img.copy()

        dst=a.out_root/r["path"]
        dst.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(dst),clean,[int(cv2.IMWRITE_JPEG_QUALITY),95])

        fo=features(img); fc=features(clean)
        rec={
            "path":r["path"],"label":r["label"],"video_id":r["video_id"],
            "frame_num":r["frame_num"],"split":r["split"],
            "mask_fraction":cov,
            "source_sha256":r["sha256"],
            "clean_sha256":sha256_file(dst),
            "source_bytes":src.stat().st_size,
            "clean_bytes":dst.stat().st_size,
        }
        manifest.append(rec)
        feats.append({
            **{k:rec[k] for k in ["path","label","video_id","frame_num","split","mask_fraction"]},
            **{f"original_{k}":v for k,v in fo.items()},
            **{f"cleaned_{k}":v for k,v in fc.items()},
            **{f"delta_{k}":fc[k]-fo[k] for k in fo},
            "original_jpeg_bytes":src.stat().st_size,
            "cleaned_jpeg_bytes":dst.stat().st_size,
            "delta_jpeg_bytes":dst.stat().st_size-src.stat().st_size,
        })

        if len(preview)<8 and cov>0.01:
            preview.append((Path(r["path"]).name,img.copy(),clean.copy(),cov))

        if (idx+1)%250==0:
            print(f"processed {idx+1}/{len(rows)}",flush=True)

    def write_csv(path,data):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    write_csv(a.report_dir/"hand_removed_manifest.csv",manifest)
    write_csv(a.report_dir/"paired_visual_features.csv",feats)

    metrics=["brightness_mean","contrast_std","sharpness_laplacian_var","saturation_mean","edge_density"]
    summary={
        "image_count":len(rows),
        "images_with_nonzero_mask":detected,
        "images_without_detected_border_skin":len(rows)-detected,
        "dimensions_preserved":"yes; outputs preserve each source image dimensions",
        "method":"border-connected skin-region detection in YCrCb+HSV, morphological cleanup/dilation, half-resolution Telea inpainting, feathered composite",
        "purpose":"derived sensitivity/ablation dataset only; canonical originals remain authoritative",
        "mask_fraction":{
            "mean":float(np.mean([x["mask_fraction"] for x in manifest])),
            "median":float(np.median([x["mask_fraction"] for x in manifest])),
            "max":float(np.max([x["mask_fraction"] for x in manifest])),
        },
        "feature_stats":{}
    }
    for m in metrics:
        o=np.array([x[f"original_{m}"] for x in feats],float)
        c=np.array([x[f"cleaned_{m}"] for x in feats],float)
        d=c-o
        summary["feature_stats"][m]={
            "original":{"mean":float(o.mean()),"std":float(o.std()),"min":float(o.min()),"max":float(o.max())},
            "cleaned":{"mean":float(c.mean()),"std":float(c.std()),"min":float(c.min()),"max":float(c.max())},
            "paired_delta":{"mean":float(d.mean()),"std":float(d.std()),"median":float(np.median(d)),"min":float(d.min()),"max":float(d.max())},
        }
    ob=np.array([x["original_jpeg_bytes"] for x in feats],float)
    cb=np.array([x["cleaned_jpeg_bytes"] for x in feats],float)
    summary["feature_stats"]["jpeg_bytes"]={
        "original":{"mean":float(ob.mean()),"std":float(ob.std()),"min":float(ob.min()),"max":float(ob.max())},
        "cleaned":{"mean":float(cb.mean()),"std":float(cb.std()),"min":float(cb.min()),"max":float(cb.max())},
        "paired_delta":{"mean":float((cb-ob).mean()),"std":float((cb-ob).std()),"median":float(np.median(cb-ob)),"min":float((cb-ob).min()),"max":float((cb-ob).max())},
    }

    (a.report_dir/"visual_feature_comparison_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    make_sheet(preview,a.report_dir/"before_after_preview_sheet.jpg")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
