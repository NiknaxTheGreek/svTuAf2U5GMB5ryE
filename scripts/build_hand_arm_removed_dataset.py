#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

HANDS = mp.solutions.hands
POSE = mp.solutions.pose

def feat(arr_bgr: np.ndarray):
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    h = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = h / max(h.sum(), 1)
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

def skin_mask(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ycrcb, np.array([0,133,77],np.uint8), np.array([255,173,127],np.uint8))
    # broad warm-skin guard, intentionally permissive but used only near detected limbs
    m2 = cv2.inRange(hsv, np.array([0,18,30],np.uint8), np.array([35,255,255],np.uint8))
    m = cv2.bitwise_and(m1,m2)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=2)

def normpt(lm, w, h):
    return (int(np.clip(lm.x,0,1)*(w-1)), int(np.clip(lm.y,0,1)*(h-1)))

def build_mask(bgr, hand_model, pose_model):
    H,W = bgr.shape[:2]
    scale = min(1.0, 720.0/max(H,W))
    small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    hres = hand_model.process(rgb)
    pres = pose_model.process(rgb)

    coarse = np.zeros((H,W), np.uint8)
    hand_count = 0
    arm_segments = 0

    if hres.multi_hand_landmarks:
        for hand in hres.multi_hand_landmarks:
            pts = []
            for lm in hand.landmark:
                x = int(np.clip(lm.x,0,1)*(W-1))
                y = int(np.clip(lm.y,0,1)*(H-1))
                pts.append([x,y])
            pts=np.asarray(pts,np.int32)
            hull=cv2.convexHull(pts)
            cv2.fillConvexPoly(coarse,hull,255)
            hand_count += 1

    if pres.pose_landmarks:
        lms = pres.pose_landmarks.landmark
        # shoulder->elbow->wrist for both arms
        for s,e,wri in [(11,13,15),(12,14,16)]:
            ls,le,lw=lms[s],lms[e],lms[wri]
            if max(ls.visibility,le.visibility,lw.visibility) < 0.35:
                continue
            ps,pe,pw = normpt(ls,W,H),normpt(le,W,H),normpt(lw,W,H)
            # width proportional to image width, conservative
            cv2.line(coarse,ps,pe,255,max(26,int(W*0.055)))
            cv2.line(coarse,pe,pw,255,max(22,int(W*0.045)))
            cv2.circle(coarse,pw,max(24,int(W*0.035)),255,-1)
            arm_segments += 1

    if hand_count==0 and arm_segments==0:
        return coarse, hand_count, arm_segments

    # Expand coarse detection then intersect with skin, while always keeping central hand hull core.
    expand = cv2.dilate(coarse, np.ones((max(9,int(W*0.018))|1,)*2,np.uint8), iterations=1)
    skin = skin_mask(bgr)
    refined = cv2.bitwise_and(expand, skin)

    # retain coarse hand/arm core to avoid fragmented masks
    refined = cv2.bitwise_or(refined, cv2.erode(coarse, np.ones((5,5),np.uint8), iterations=1))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((11,11),np.uint8), iterations=2)
    refined = cv2.dilate(refined, np.ones((9,9),np.uint8), iterations=1)
    return refined, hand_count, arm_segments

def clean_one(bgr, mask):
    if np.count_nonzero(mask)==0:
        return bgr.copy()
    # Telea inpainting; deterministic for fixed image/mask.
    return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)

def summarize(vals):
    a=np.asarray(vals,dtype=float)
    if len(a)==0: return {}
    return {
        "n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "q1": float(np.quantile(a,.25)),
        "median": float(np.median(a)), "q3": float(np.quantile(a,.75)),
        "max": float(a.max())
    }

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

    manifests=[]
    features=[]
    qa=[]
    with HANDS.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.35) as hand_model, \
         POSE.Pose(static_image_mode=True,model_complexity=1,enable_segmentation=False,min_detection_confidence=0.35) as pose_model:
        for idx,r in enumerate(rows):
            src=args.image_root/r["path"]
            if not src.exists(): src=args.image_root/"images"/r["path"]
            bgr=cv2.imread(str(src),cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read {src}")
            H,W=bgr.shape[:2]
            mask,hc,ac=build_mask(bgr,hand_model,pose_model)
            cleaned=clean_one(bgr,mask)

            dst=args.clean_root/r["path"]
            dst.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dst),cleaned,[int(cv2.IMWRITE_JPEG_QUALITY),95])

            cov=float((mask>0).mean())
            fo=feat(bgr); fc=feat(cleaned)
            rec={
                "path":r["path"],"label":r["label"],"split":r["split"],
                "video_id":r["video_id"],"frame_num":r["frame_num"],
                "width":W,"height":H,"hand_count":hc,"arm_segments":ac,
                "mask_fraction":cov,"changed":int(cov>0)
            }
            manifests.append(rec)
            features.append({
                **rec,
                **{f"orig_{k}":v for k,v in fo.items()},
                **{f"clean_{k}":v for k,v in fc.items()},
                **{f"delta_{k}":fc[k]-fo[k] for k in fo}
            })
            if cov>0:
                qa.append((cov,r["path"],str(src),str(dst),mask.copy()))
            if (idx+1)%100==0:
                print(f"processed {idx+1}/{len(rows)}", flush=True)

    # CSV outputs
    def write_csv(path,data):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
    write_csv(args.out_dir/"cleaned_manifest.csv",manifests)
    write_csv(args.out_dir/"visual_features_original_vs_cleaned.csv",features)

    # summary stats overall + labels
    keys=["brightness_mean","contrast_std","sharpness_laplacian_var","entropy_bits","edge_density"]
    summary={
        "image_count":len(rows),
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
        "dimensions": {
            "unique_widths": sorted(set(int(x["width"]) for x in manifests)),
            "unique_heights": sorted(set(int(x["height"]) for x in manifests))
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
    (args.out_dir/"hand_arm_cleaning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # QA contact sheet: 12 largest masks + 12 around median nonzero
    qa=sorted(qa,reverse=True,key=lambda x:x[0])
    picks=qa[:12]
    if len(qa)>24:
        mid=len(qa)//2
        picks += qa[max(0,mid-6):mid+6]
    cell=(216,384)
    sheet=Image.new("RGB",(4*cell[0],math.ceil(len(picks)/2)*cell[1]),"white")
    # two columns per example: before/after; 2 examples across
    rowsheet=math.ceil(len(picks)/2)
    sheet=Image.new("RGB",(4*cell[0],rowsheet*cell[1]),"white")
    for k,(_,rel,src,dst,_) in enumerate(picks):
        rr=k//2; cc=(k%2)*2
        for offset,p in [(0,src),(1,dst)]:
            with Image.open(p) as im:
                t=ImageOps.fit(im.convert("RGB"),cell,method=Image.Resampling.BILINEAR)
            sheet.paste(t,((cc+offset)*cell[0],rr*cell[1]))
    sheet.save(args.out_dir/"hand_arm_cleaning_qa.jpg",quality=90)

if __name__=="__main__":
    main()
