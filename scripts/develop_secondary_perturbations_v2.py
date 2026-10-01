from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from src.scratch_v2 import preprocess_to_uint8, read_csv, sha256_file
from src.secondary_perturbations_v2 import (
    grayscale_luma_mae,
    mask_summary,
    overlay_mask,
    pil_l_grayscale,
    rec709_grayscale,
)
from src.mediapipe_hand_mask_v2 import MediaPipeHandMasker

EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_ST_SPLIT_SHA256 = "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85"
DEV_SEED = "monreader-secondary-dev-v2"
PER_STRATUM = 6
THUMB_W, THUMB_H = 112, 199
CONFIDENCES = (0.35, 0.40, 0.50)
CANDIDATE_NAMES = ("mp_seed_c35", "mp_seed_c40", "mp_seed_c50", "mp_seed_c35_fallback")


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{DEV_SEED}|{sample_id}".encode()).hexdigest()


def select_dev_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["role"] != "context":
            continue
        strata[(row["role_detail"], row["label"], row["environment_id"])].append(row)

    chosen: list[dict[str, str]] = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: stable_key(r["sample_id"]))
        if len(bucket) < PER_STRATUM:
            raise ValueError(f"Stratum {key} has only {len(bucket)} rows")
        chosen.extend(bucket[:PER_STRATUM])

    return sorted(
        chosen,
        key=lambda r: (r["role_detail"], r["environment_id"], r["label"], r["sample_id"]),
    )


def chw_to_pil(chw: np.ndarray) -> Image.Image:
    return Image.fromarray(np.transpose(chw, (1, 2, 0)), mode="RGB")


def thumb(chw: np.ndarray) -> Image.Image:
    return chw_to_pil(chw).resize((THUMB_W, THUMB_H), Image.Resampling.BILINEAR)


def build_sheet(rows: list[tuple[str, list[tuple[str, np.ndarray]]]], path: Path) -> None:
    if not rows:
        return
    cols = max(len(panels) for _, panels in rows)
    row_h = THUMB_H + 42
    canvas = Image.new("RGB", (cols * THUMB_W, len(rows) * row_h), "white")
    draw = ImageDraw.Draw(canvas)

    for r, (sample_id, panels) in enumerate(rows):
        y0 = r * row_h
        draw.text((3, y0 + 2), sample_id[-30:], fill="black")
        for c, (title, chw) in enumerate(panels):
            x0 = c * THUMB_W
            draw.text((x0 + 2, y0 + 18), title[:19], fill="black")
            canvas.paste(thumb(chw), (x0, y0 + 40))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)

    review = canvas.resize(
        (max(1, canvas.width * 3 // 4), max(1, canvas.height * 3 // 4)),
        Image.Resampling.LANCZOS,
    )
    buffer = io.BytesIO()
    review.save(buffer, format="JPEG", quality=45, optimize=True, subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buffer.getvalue()).decode("ascii") + "\n",
        encoding="ascii",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--membership", required=True, type=Path)
    p.add_argument("--hand-model", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive hash mismatch")
    if sha256_file(args.membership) != EXPECTED_ST_SPLIT_SHA256:
        raise ValueError("ST split hash mismatch")
    if not args.hand_model.is_file():
        raise ValueError("Hand landmarker model missing")

    import mediapipe as mp

    manifest = read_csv(args.manifest)
    split = read_csv(args.membership)
    by_id = {r["sample_id"]: r for r in manifest}
    dev = select_dev_rows(split)
    if len(dev) != 48:
        raise ValueError(f"Expected 48 dev images, got {len(dev)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    sheets: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    with ExitStack() as stack:
        maskers = {
            conf: stack.enter_context(
                MediaPipeHandMasker(
                    args.hand_model,
                    min_hand_detection_confidence=conf,
                    min_hand_presence_confidence=conf,
                    num_hands=2,
                )
            )
            for conf in CONFIDENCES
        }
        z = stack.enter_context(ZipFile(args.archive))

        for row in dev:
            source = by_id[row["sample_id"]]
            with z.open(source["archive_member"]) as fh:
                with Image.open(fh) as image:
                    original = preprocess_to_uint8(image)

            gray709 = rec709_grayscale(original)
            graypil = pil_l_grayscale(original)

            per_conf: dict[float, tuple[dict[str, np.ndarray], int]] = {}
            for conf, masker in maskers.items():
                per_conf[conf] = masker.candidate_masks(original)

            masks = {
                "mp_seed_c35": per_conf[0.35][0]["mp_seed_skin"],
                "mp_seed_c40": per_conf[0.40][0]["mp_seed_skin"],
                "mp_seed_c50": per_conf[0.50][0]["mp_seed_skin"],
                "mp_seed_c35_fallback": per_conf[0.35][0]["mp_seed_skin_fallback"],
            }

            entry = {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "environment_id": row["environment_id"],
                "role_detail": row["role_detail"],
                "detected_hands": {
                    f"{conf:.2f}": int(per_conf[conf][1])
                    for conf in CONFIDENCES
                },
                "gray_rec709_luma_mae": grayscale_luma_mae(original, gray709),
                "gray_pil_l_luma_mae": grayscale_luma_mae(original, graypil),
                "hand_candidates": {
                    name: mask_summary(mask)
                    for name, mask in masks.items()
                },
            }

            panels = [("original", original), ("gray_rec709", gray709)]
            panels.extend((name, overlay_mask(original, masks[name])) for name in CANDIDATE_NAMES)
            metrics.append(entry)
            sheets.append((row["sample_id"], panels))

    for start in range(0, len(sheets), 12):
        build_sheet(
            sheets[start:start + 12],
            args.output_dir / f"hand_mask_candidates_{start // 12 + 1}.png",
        )

    aggregate: dict[str, dict] = {}
    for name in CANDIDATE_NAMES:
        vals = [m["hand_candidates"][name]["coverage_fraction"] for m in metrics]
        nonzero = [v for v in vals if v > 0]
        aggregate[name] = {
            "images": len(vals),
            "nonzero_masks": len(nonzero),
            "nonzero_fraction": len(nonzero) / len(vals),
            "mean_coverage": float(np.mean(vals)),
            "median_coverage": float(np.median(vals)),
            "p90_coverage": float(np.quantile(vals, 0.90)),
            "max_coverage": float(np.max(vals)),
        }

    detector_stats: dict[str, dict] = {}
    for conf in CONFIDENCES:
        key = f"{conf:.2f}"
        counts = [int(m["detected_hands"][key]) for m in metrics]
        detector_stats[key] = {
            "images_with_detection": int(sum(x > 0 for x in counts)),
            "images_without_detection": int(sum(x == 0 for x in counts)),
            "one_hand_images": int(sum(x == 1 for x in counts)),
            "two_hand_images": int(sum(x >= 2 for x in counts)),
        }

    summary = {
        "status": "SECONDARY_PERTURBATION_DEV_ONLY",
        "primary_test_images_used": 0,
        "development_population": "fixed stratified sample from ST context-only rows",
        "development_images": len(dev),
        "selection_rule": "visual mask quality + mask-coverage sanity only; never classifier performance",
        "grayscale": {
            "candidate_rec709_mean_luma_mae": float(
                np.mean([m["gray_rec709_luma_mae"] for m in metrics])
            ),
            "candidate_pil_l_mean_luma_mae": float(
                np.mean([m["gray_pil_l_luma_mae"] for m in metrics])
            ),
            "recommended_before_visual_review": "rec709_grayscale",
        },
        "hand_detector": {
            "library": "mediapipe",
            "library_version": str(mp.__version__),
            "model_sha256": sha256_file(args.hand_model),
            "num_hands": 2,
            "confidence_comparison": detector_stats,
        },
        "hand_mask_candidates": aggregate,
        "hand_mask_status": "REQUIRES_VISUAL_REVIEW_BEFORE_FREEZE",
        "dev_sample_ids": [r["sample_id"] for r in dev],
    }

    (args.output_dir / "development_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "development_per_image.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = ["sample_id", "role_detail", "environment_id", "label"]
    with (args.output_dir / "development_sample.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in dev])

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
