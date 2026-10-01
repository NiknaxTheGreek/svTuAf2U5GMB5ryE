from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def build_sheet(rows, image_root: Path, out_root: Path, path: Path):
    cell=(216,384); gap=10; label_h=42
    canvas=Image.new("RGB",(2*cell[0]+3*gap,len(rows)*(cell[1]+label_h+gap)+gap),"white")
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(rows):
        src=image_root/r["path"]
        dst=out_root/r["path"]
        y=gap+i*(cell[1]+label_h+gap)
        for col,p in enumerate((src,dst)):
            with Image.open(p) as im:
                thumb=ImageOps.fit(im.convert("RGB"),cell,method=Image.Resampling.BILINEAR)
            x=gap+col*(cell[0]+gap)
            canvas.paste(thumb,(x,y))
        draw.text((gap,y+cell[1]+5),f'BEFORE {r["label"]} {r["video_id"]} f{r["frame_num"]}',fill="black")
        draw.text((2*gap+cell[0],y+cell[1]+5),f'AFTER mask={float(r["mask_fraction"]):.3f}',fill="black")
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=94)
    buf=io.BytesIO()
    canvas.resize((canvas.width*3//4,canvas.height*3//4),Image.Resampling.LANCZOS).save(
        buf,format="JPEG",quality=60,optimize=True,subsampling=2
    )
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii")+"\n",encoding="ascii"
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",required=True,type=Path)
    ap.add_argument("--image-root",required=True,type=Path)
    ap.add_argument("--clean-root",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()
    with args.report.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=48: raise ValueError("Expected 48 cleaning rows")

    for start in range(0,48,12):
        build_sheet(rows[start:start+12],args.image_root,args.clean_root,args.output_dir/f"historical_inpainting_{start//12+1}.jpg")

    vals=np.asarray([float(r["mask_fraction"]) for r in rows])
    arm=np.asarray([int(r["arm_pixels"]) for r in rows])
    hand=np.asarray([int(r["hand_pixels"]) for r in rows])
    summary={
        "status":"HISTORICAL_SCHP_MEDIAPIPE_TELEA_48_READY_FOR_VISUAL_REVIEW",
        "images":48,
        "zero_combined_mask_images":int(np.sum(vals==0)),
        "zero_arm_mask_images":int(np.sum(arm==0)),
        "zero_hand_mask_images":int(np.sum(hand==0)),
        "mean_mask_fraction":float(vals.mean()),
        "median_mask_fraction":float(np.median(vals)),
        "p90_mask_fraction":float(np.quantile(vals,0.9)),
        "max_mask_fraction":float(vals.max()),
        "classifier_predictions_used":False,
        "algorithm_source_commit":"0b06f7163998c06135c6ea18d286f02fcedb9caa",
    }
    (args.output_dir/"qa_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
