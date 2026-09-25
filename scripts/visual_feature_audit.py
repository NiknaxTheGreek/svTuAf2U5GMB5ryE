#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def entropy_u8(gray):
    hist=np.bincount(gray.reshape(-1),minlength=256).astype(np.float64)
    p=hist/hist.sum()
    p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def features(path: Path):
    data=np.fromfile(path,dtype=np.uint8)
    bgr=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unreadable {path}")
    h,w=bgr.shape[:2]
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    # downsample only for expensive edge calculation; global moments are full image.
    small=cv2.resize(gray,(270,480),interpolation=cv2.INTER_AREA)
    edges=cv2.Canny(small,80,160)
    lap=cv2.Laplacian(small,cv2.CV_64F)
    return {
      "width":int(w),"height":int(h),"channels":int(bgr.shape[2]),
      "brightness_mean":float(gray.mean()/255.0),
      "contrast_std":float(gray.std()/255.0),
      "saturation_mean":float(hsv[:,:,1].mean()/255.0),
      "sharpness_laplacian_var":float(lap.var()),
      "edge_density":float((edges>0).mean()),
      "entropy_bits":entropy_u8(gray),
      "jpeg_bytes":int(path.stat().st_size),
    }

def summarise(vals):
    a=np.asarray(vals,dtype=float)
    return {
      "min":float(np.min(a)),
      "q1":float(np.quantile(a,.25)),
      "median":float(np.median(a)),
      "mean":float(np.mean(a)),
      "q3":float(np.quantile(a,.75)),
      "max":float(np.max(a)),
      "std":float(np.std(a)),
      "unique_rounded_6dp":int(len(set(np.round(a,6).tolist())))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--dataset-name",required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    with args.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))

    out=[]
    for i,r in enumerate(rows,1):
        p=args.image_root/r["path"]
        if not p.exists():
            p=args.image_root/"images"/r["path"]
        z=features(p)
        out.append({
          "dataset":args.dataset_name,
          "path":r["path"],"label":r["label"],"split":r["split"],
          "video_id":r["video_id"],"frame_num":int(r["frame_num"]),**z
        })
        if i%250==0 or i==len(rows): print(i,len(rows))

    with (args.out_dir/f"{args.dataset_name}_visual_features.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    numeric=["width","height","channels","brightness_mean","contrast_std","saturation_mean",
             "sharpness_laplacian_var","edge_density","entropy_bits","jpeg_bytes"]
    summary={
      "dataset":args.dataset_name,
      "image_count":len(out),
      "features":{k:summarise([r[k] for r in out]) for k in numeric},
      "by_label":{
        label:{k:summarise([r[k] for r in out if r["label"]==label]) for k in numeric}
        for label in sorted(set(r["label"] for r in out))
      },
      "interpretation_note":"Width/height/channels can be fixed while photometric/texture/compression properties vary. Edge density is a simple structure/text proxy, not literal OCR text density."
    }
    (args.out_dir/f"{args.dataset_name}_visual_feature_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
