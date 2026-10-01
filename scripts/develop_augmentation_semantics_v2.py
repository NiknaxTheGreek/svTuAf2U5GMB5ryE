from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.scratch_v2 import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    preprocess_to_uint8,
    read_csv,
    sha256_file,
)
from src.secondary_augmentation_v2 import (
    apply_params,
    load_augmentation_config,
    sample_params,
    stable_sample_generator,
)


THUMB_W, THUMB_H = 112, 199


def chw_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (
        x.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")


def build_sheet(rows, path: Path) -> None:
    row_h = THUMB_H + 42
    canvas = Image.new("RGB", (2 * THUMB_W, len(rows) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (sample_id, original, augmented) in enumerate(rows):
        y0 = i * row_h
        draw.text((3, y0 + 2), sample_id[-30:], fill="black")
        for c, (title, img) in enumerate((("original", original), ("augmented", augmented))):
            x0 = c * THUMB_W
            draw.text((x0 + 2, y0 + 18), title, fill="black")
            thumb = chw_to_pil(img).resize((THUMB_W, THUMB_H), Image.Resampling.BILINEAR)
            canvas.paste(thumb, (x0, y0 + 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    buf = io.BytesIO()
    canvas.resize(
        (max(1, canvas.width * 3 // 4), max(1, canvas.height * 3 // 4)),
        Image.Resampling.LANCZOS,
    ).save(buf, format="JPEG", quality=50, optimize=True, subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii") + "\n",
        encoding="ascii",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--membership", required=True, type=Path)
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive SHA mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Manifest SHA mismatch")
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("ST split SHA mismatch")

    cfg = load_augmentation_config(args.config)
    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    with args.sample.open(newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    if len(sample) != 48:
        raise ValueError("Expected fixed 48-image development sample")

    split = read_csv(args.membership)
    primary_test = {r["sample_id"] for r in split if r["role"] == "test"}
    if any(r["sample_id"] in primary_test for r in sample):
        raise ValueError("Primary ST test image entered augmentation semantic gate")

    stats = []
    sheets = []
    with ZipFile(args.archive) as z:
        for row in sample:
            src = by_id[row["sample_id"]]
            with z.open(src["archive_member"]) as fh:
                with Image.open(fh) as im:
                    arr = preprocess_to_uint8(im)
            original = torch.from_numpy(arr).float().div_(255.0)
            g = stable_sample_generator(row["sample_id"])
            params = sample_params(original.shape[1], original.shape[2], cfg, g)
            augmented = apply_params(original.clone(), params)
            stats.append({"sample_id": row["sample_id"], **params})
            sheets.append((row["sample_id"], original, augmented))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, 48, 12):
        build_sheet(sheets[start:start+12], args.output_dir / f"augmentation_semantic_{start//12+1}.png")

    summary = {
        "status": "AUGMENTATION_SEMANTIC_GATE_AWAITING_VISUAL_REVIEW",
        "primary_test_images_used": 0,
        "development_images": 48,
        "config_sha256": "5168b13d3ce4f286df71a1f823cb78889098fa843602154c34683ede80dce18b",
        "observed_parameters": {
            "angle_min": float(min(x["angle_degrees"] for x in stats)),
            "angle_max": float(max(x["angle_degrees"] for x in stats)),
            "translate_x_min": int(min(x["translate_x_pixels"] for x in stats)),
            "translate_x_max": int(max(x["translate_x_pixels"] for x in stats)),
            "translate_y_min": int(min(x["translate_y_pixels"] for x in stats)),
            "translate_y_max": int(max(x["translate_y_pixels"] for x in stats)),
            "scale_min": float(min(x["scale"] for x in stats)),
            "scale_max": float(max(x["scale"] for x in stats)),
            "brightness_min": float(min(x["brightness_factor"] for x in stats)),
            "brightness_max": float(max(x["brightness_factor"] for x in stats)),
            "contrast_min": float(min(x["contrast_factor"] for x in stats)),
            "contrast_max": float(max(x["contrast_factor"] for x in stats)),
        },
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (args.output_dir/"params.json").write_text(json.dumps(stats, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
