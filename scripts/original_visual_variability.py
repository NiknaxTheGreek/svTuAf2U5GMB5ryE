#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np

SIZE=(256,455)

def feats(p:Path):
    im=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if im is None: raise RuntimeError(f"Unreadable: {p}")
    im=cv2.resize(im,SIZE,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    return {
        "brightness_mean":float(gray.mean()),
        "contrast_std":float(gray.std()),
        "sharpness_laplacian_var":float(cv2.Laplacian(gray,cv2.CV_64F).var()),
        "saturation_mean":float(hsv[:,:,1].mean()),
        "edge_density":float((cv2.Canny(gray,80,160)>0).mean()),
        "jpeg_bytes":float(p.stat().st_size),
    }

def grade(metric,stats):
    cv=stats["cv"]
    iqrm=stats["iqr_over_median"]
    p90m=stats["p95_p5_over_median"]
    # Sharpness and edge measures are naturally skewed; use robust spread more heavily.
    if metric=="sharpness_laplacian_var":
        score=max(iqrm,p90m/2)
    elif metric in {"edge_density","saturation_mean"}:
        score=max(cv,iqrm,p90m/2)
    else:
        score=max(cv,iqrm)
    if score < 0.05: return "negligible"
    if score < 0.15: return "low-moderate"
    if score < 0.35: return "substantial"
    return "extreme"

def summarize(v):
    a=np.asarray(v,dtype=float)
    q5,q25,q50,q75,q95=np.quantile(a,[.05,.25,.5,.75,.95])
    mean=float(a.mean()); sd=float(a.std(ddof=0))
    med=float(q50); iqr=float(q75-q25)
    return {
        "mean":mean,"std":sd,"cv":float(sd/mean) if abs(mean)>1e-12 else None,
        "min":float(a.min()),"p05":float(q5),"p25":float(q25),"median":med,
        "p75":float(q75),"p95":float(q95),"max":float(a.max()),
        "iqr":iqr,
        "iqr_over_median":float(iqr/med) if abs(med)>1e-12 else None,
        "p95_p5_over_median":float((q95-q5)/med) if abs(med)>1e-12 else None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    with a.inventory.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    out=[]
    for i,r in enumerate(rows,1):
        p=a.image_root/r["path"]
        if not p.exists(): p=a.image_root/"images"/r["path"]
        rec={"path":r["path"],"label":r["label"],"split":r["split"],"video_id":r["video_id"],"frame_num":r["frame_num"]}
        rec.update(feats(p)); out.append(rec)
        if i%500==0: print(f"processed {i}/{len(rows)}",flush=True)

    metrics=["brightness_mean","contrast_std","sharpness_laplacian_var","saturation_mean","edge_density","jpeg_bytes"]
    summary={"image_count":len(out),"analysis_resolution":"256x455 for pixel-derived features; jpeg_bytes from original file","metrics":{}}
    for m in metrics:
        st=summarize([r[m] for r in out]); st["variability_grade"]=grade(m,st)
        summary["metrics"][m]=st

    with (a.out_dir/"original_visual_features.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    (a.out_dir/"original_variability_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
