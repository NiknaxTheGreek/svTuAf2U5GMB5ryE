from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from src.scratch_v2 import preprocess_to_uint8, read_csv
from src.secondary_color_v3 import linear_srgb_grayscale
from src.secondary_perturbations_v2 import apply_hand_mask, overlay_mask
from src.hand_mask_v3 import GEOMETRY_CANDIDATES, MultiViewMediaPipeHandMasker, geometry_mask

EXAMPLES = [
    "testing/flip/0048_000000011",
    "testing/notflip/0040_000000004",
    "testing/flip/0008_000000029",
    "training/notflip/0016_000000023",
    "testing/flip/0059_000000028",
    "training/flip/0018_000000021",
]

PANEL_W, PANEL_H = 224, 398

def chw_to_pil(chw: np.ndarray) -> Image.Image:
    return Image.fromarray(np.transpose(chw, (1,2,0)), mode="RGB")

def diff_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.transpose(a, (1,2,0)).astype(np.int16)
    y = np.transpose(b, (1,2,0)).astype(np.int16)
    d = np.max(np.abs(x-y), axis=2).astype(np.uint8)
    heat = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    heat[...,0] = d
    return np.transpose(heat, (2,0,1))

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--hand-model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args=p.parse_args()

    manifest={r["sample_id"]:r for r in read_csv(args.manifest)}
    cfg={c.name:c for c in GEOMETRY_CANDIDATES}["mv35_handarm"]

    rows=[]
    with MultiViewMediaPipeHandMasker(args.hand_model,0.35) as masker, ZipFile(args.archive) as z:
        for sample_id in EXAMPLES:
            src=manifest[sample_id]
            with z.open(src["archive_member"]) as fh:
                with Image.open(fh) as im:
                    source_rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
                    original=preprocess_to_uint8(im)
            _,h,w=original.shape
            detections,_=masker.detections(source_rgb,h,w)
            mask=geometry_mask(detections,h,w,cfg)
            gray=linear_srgb_grayscale(original)
            removed=apply_hand_mask(original,mask,fill="local_border_median")
            overlay=overlay_mask(original,mask)
            diff=diff_map(original,removed)
            rows.append((sample_id,[
                ("ORIGINAL", original),
                ("GRAYSCALE", gray),
                ("MASK OVERLAY", overlay),
                ("HAND REMOVED", removed),
                ("DIFFERENCE", diff),
            ]))

    header_h=34
    row_h=PANEL_H+header_h+24
    canvas=Image.new("RGB",(5*PANEL_W,len(rows)*row_h),"white")
    draw=ImageDraw.Draw(canvas)
    for ri,(sample_id,panels) in enumerate(rows):
        y0=ri*row_h
        draw.text((4,y0+4),sample_id,fill="black")
        for ci,(title,chw) in enumerate(panels):
            x0=ci*PANEL_W
            draw.text((x0+4,y0+18),title,fill="black")
            canvas.paste(chw_to_pil(chw),(x0,y0+header_h))

    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output,optimize=True)

    review=canvas.resize((canvas.width//2, canvas.height//2),Image.Resampling.LANCZOS)
    buf=io.BytesIO()
    review.save(buf,format="JPEG",quality=72,optimize=True)
    args.output.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii")+"\n",encoding="ascii"
    )
    return 0

if __name__=="__main__":
    raise SystemExit(main())
