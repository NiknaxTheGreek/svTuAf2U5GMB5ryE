from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from src.scratch_v2 import preprocess_to_uint8, read_csv, sha256_file
from src.secondary_color_v3 import (
    chroma_residual_max,
    linear_luminance_error,
    linear_srgb_grayscale,
)
from src.secondary_perturbations_v2 import (
    mask_summary,
    overlay_mask,
    rec709_grayscale,
)
from src.hand_mask_v3 import (
    GEOMETRY_CANDIDATES,
    MultiViewMediaPipeHandMasker,
    geometry_mask,
)

EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_ST_SPLIT_SHA256 = "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85"
DEV_SEED = "monreader-secondary-dev-v2"
PER_STRATUM = 6
THUMB_W, THUMB_H = 112, 199


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{DEV_SEED}|{sample_id}".encode()).hexdigest()


def select_dev_rows(rows: list[dict[str,str]]) -> list[dict[str,str]]:
    strata: dict[tuple[str,str,str], list[dict[str,str]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "context":
            strata[(row["role_detail"], row["label"], row["environment_id"])].append(row)
    chosen = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: stable_key(r["sample_id"]))
        if len(bucket) < PER_STRATUM:
            raise ValueError(f"Stratum {key} has only {len(bucket)} rows")
        chosen.extend(bucket[:PER_STRATUM])
    return sorted(
        chosen,
        key=lambda r:(r["role_detail"],r["environment_id"],r["label"],r["sample_id"]),
    )


def chw_to_pil(chw: np.ndarray) -> Image.Image:
    return Image.fromarray(np.transpose(chw,(1,2,0)), mode="RGB")


def thumb(chw: np.ndarray) -> Image.Image:
    return chw_to_pil(chw).resize((THUMB_W,THUMB_H), Image.Resampling.BILINEAR)


def build_sheet(rows, path: Path) -> None:
    cols = max(len(panels) for _,panels in rows)
    row_h = THUMB_H + 42
    canvas = Image.new("RGB",(cols*THUMB_W,len(rows)*row_h),"white")
    draw = ImageDraw.Draw(canvas)
    for r,(sample_id,panels) in enumerate(rows):
        y0=r*row_h
        draw.text((3,y0+2),sample_id[-30:],fill="black")
        for c,(title,chw) in enumerate(panels):
            x0=c*THUMB_W
            draw.text((x0+2,y0+18),title[:19],fill="black")
            canvas.paste(thumb(chw),(x0,y0+40))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path,optimize=True)
    review=canvas.resize(
        (max(1,canvas.width*3//4),max(1,canvas.height*3//4)),
        Image.Resampling.LANCZOS,
    )
    buf=io.BytesIO()
    review.save(buf,format="JPEG",quality=48,optimize=True,subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii")+"\n",
        encoding="ascii",
    )


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--membership",required=True,type=Path)
    p.add_argument("--hand-model",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args()

    if sha256_file(args.archive)!=EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive hash mismatch")
    if sha256_file(args.membership)!=EXPECTED_ST_SPLIT_SHA256:
        raise ValueError("ST split hash mismatch")

    manifest=read_csv(args.manifest)
    split=read_csv(args.membership)
    by_id={r["sample_id"]:r for r in manifest}
    dev=select_dev_rows(split)
    if len(dev)!=48:
        raise ValueError(f"Expected 48 development images, got {len(dev)}")

    cfg_by_name={c.name:c for c in GEOMETRY_CANDIDATES}
    names=("mv50_hull","mv50_handarm","mv35_handarm")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)

    metrics=[]
    sheets=[]
    with MultiViewMediaPipeHandMasker(args.hand_model,0.50) as mp50,          MultiViewMediaPipeHandMasker(args.hand_model,0.35) as mp35,          ZipFile(args.archive) as z:
        for row in dev:
            source=by_id[row["sample_id"]]
            with z.open(source["archive_member"]) as fh:
                with Image.open(fh) as im:
                    source_rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
                    original=preprocess_to_uint8(im)

            gray_linear=linear_srgb_grayscale(original)
            gray_gamma=rec709_grayscale(original)
            err_linear=linear_luminance_error(original,gray_linear)
            err_gamma=linear_luminance_error(original,gray_gamma)

            _,h,w=original.shape
            det50,views50=mp50.detections(source_rgb,h,w)
            det35,views35=mp35.detections(source_rgb,h,w)
            masks={
                "mv50_hull":geometry_mask(det50,h,w,cfg_by_name["mv50_hull"]),
                "mv50_handarm":geometry_mask(det50,h,w,cfg_by_name["mv50_handarm"]),
                "mv35_handarm":geometry_mask(det35,h,w,cfg_by_name["mv35_handarm"]),
            }

            metrics.append({
                "sample_id":row["sample_id"],
                "label":row["label"],
                "environment_id":row["environment_id"],
                "role_detail":row["role_detail"],
                "linear_gray_luminance_error":err_linear,
                "gamma_gray_linear_luminance_error":err_gamma,
                "linear_gray_chroma_residual_max":chroma_residual_max(gray_linear),
                "detections_50":len(det50),
                "detections_35":len(det35),
                "views_50":views50,
                "views_35":views35,
                "hand_candidates":{
                    name:mask_summary(masks[name]) for name in names
                },
            })
            panels=[
                ("original",original),
                ("gray_linear",gray_linear),
                ("gray_gamma",gray_gamma),
            ]
            panels.extend((n,overlay_mask(original,masks[n])) for n in names)
            sheets.append((row["sample_id"],panels))

    for start in range(0,len(sheets),12):
        build_sheet(
            sheets[start:start+12],
            out/f"v3_candidates_{start//12+1}.png",
        )

    hand_summary={}
    for name in names:
        vals=[m["hand_candidates"][name]["coverage_fraction"] for m in metrics]
        hand_summary[name]={
            "nonzero_masks":int(sum(v>0 for v in vals)),
            "nonzero_fraction":float(np.mean(np.asarray(vals)>0)),
            "mean_coverage":float(np.mean(vals)),
            "median_coverage":float(np.median(vals)),
            "p90_coverage":float(np.quantile(vals,0.90)),
            "max_coverage":float(np.max(vals)),
        }

    linear_mae=[m["linear_gray_luminance_error"]["mean_abs"] for m in metrics]
    gamma_mae=[m["gamma_gray_linear_luminance_error"]["mean_abs"] for m in metrics]
    summary={
        "status":"SECONDARY_PERTURBATION_V3_DEV_ONLY",
        "primary_test_images_used":0,
        "development_images":48,
        "development_population":"same fixed stratified ST context-only sample as V2",
        "selection_rule":"visual fidelity/detection quality/perturbation specificity only; never classifier performance",
        "colour":{
            "linear_srgb_grayscale_mean_linear_luminance_mae":float(np.mean(linear_mae)),
            "gamma_rec709_grayscale_mean_linear_luminance_mae":float(np.mean(gamma_mae)),
            "linear_srgb_chroma_residual_max":int(max(m["linear_gray_chroma_residual_max"] for m in metrics)),
        },
        "hand_detection":{
            "images_with_detection_50":int(sum(m["detections_50"]>0 for m in metrics)),
            "images_without_detection_50":int(sum(m["detections_50"]==0 for m in metrics)),
            "images_with_detection_35":int(sum(m["detections_35"]>0 for m in metrics)),
            "images_without_detection_35":int(sum(m["detections_35"]==0 for m in metrics)),
        },
        "hand_mask_candidates":hand_summary,
        "dev_sample_ids":[r["sample_id"] for r in dev],
    }
    (out/"development_summary_v3.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    (out/"development_per_image_v3.json").write_text(
        json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    fields=["sample_id","role_detail","environment_id","label"]
    with (out/"development_sample_v3.csv").open("w",newline="",encoding="utf-8") as fh:
        wr=csv.DictWriter(fh,fieldnames=fields)
        wr.writeheader()
        wr.writerows([{k:r[k] for k in fields} for r in dev])

    print(json.dumps(summary,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
