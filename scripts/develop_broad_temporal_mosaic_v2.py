from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from scripts.hand_arm_clean_dataset_onnx_tasks_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    ONNX_FILE,
    TaskHandMasker,
)
from scripts.develop_video_assisted_inpainting_v2 import (
    read_csv,
    decode_member,
    final_hand_arm_mask,
    target_features,
    estimate_alignment,
)
from scripts.develop_lama_inpainting_v2 import run_lama_scaled

MAX_SIDE = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def best_source_mosaic(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    aligned_sources: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Copy each masked target pixel from the highest-ranked accepted real source."""
    h, w = target_mask.shape
    out = target_rgb.copy()
    filled = np.zeros((h, w), dtype=bool)

    ranked = sorted(
        aligned_sources,
        key=lambda x: (
            -(x["alignment"]["quality"] / (1.0 + 0.02 * x["frame_distance"])),
            x["frame_distance"],
            x["sample_id"],
        ),
    )

    details = []
    for cand in ranked:
        a = cand["alignment"]
        score = a["quality"] / (1.0 + 0.02 * cand["frame_distance"])
        valid = a["fill_valid"] & (~filled) & (target_mask > 0)
        n = int(valid.sum())
        if n:
            out[valid] = a["warped"][valid]
            filled[valid] = True
        details.append({
            "sample_id": cand["sample_id"],
            "frame_distance": cand["frame_distance"],
            "good_matches": a["good_matches"],
            "inliers": a["inliers"],
            "inlier_ratio": a["inlier_ratio"],
            "overlap_fraction": a["overlap_fraction"],
            "median_gray_abs_error": a["median_gray_abs_error"],
            "quality": a["quality"],
            "rank_score": score,
            "new_fill_pixels": n,
        })
    return out, filled, details


def residual_preview(mosaic: np.ndarray, target_mask: np.ndarray, filled: np.ndarray) -> np.ndarray:
    out = mosaic.copy().astype(np.float32)
    residual = (target_mask > 0) & (~filled)
    tint = np.zeros_like(out)
    tint[..., 2] = 255
    a = residual[..., None].astype(np.float32) * 0.42
    out = out * (1.0 - a) + tint * a
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_preview(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[..., 0] = 255
    a = (mask > 0)[..., None].astype(np.float32) * 0.45
    return np.clip(out * (1.0 - a) + tint * a, 0, 255).astype(np.uint8)


def build_sheet(entries: list[dict], path: Path) -> None:
    cell = (150, 267)
    gap = 7
    label_h = 50
    rows = len(entries)
    canvas = Image.new(
        "RGB",
        (4 * cell[0] + 5 * gap, rows * (cell[1] + label_h + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for i, e in enumerate(entries):
        y = gap + i * (cell[1] + label_h + gap)
        imgs = [
            e["original"],
            mask_preview(e["original"], e["mask"]),
            residual_preview(e["mosaic"], e["mask"], e["filled"]),
            e["final"],
        ]
        labels = [
            "BEFORE",
            "MASK",
            f'MOSAIC real={e["real_fraction"]:.2f}',
            "MOSAIC + LAMA RESIDUAL",
        ]
        for col, (arr, label) in enumerate(zip(imgs, labels)):
            x = gap + col * (cell[0] + gap)
            thumb = ImageOps.fit(
                Image.fromarray(arr).convert("RGB"),
                cell,
                method=Image.Resampling.BILINEAR,
            )
            canvas.paste(thumb, (x, y))
            draw.text((x, y + cell[1] + 3), label, fill="black")

        draw.text(
            (gap, y + cell[1] + 24),
            f'{e["sample_id"]} pool={e["pool"]} accepted={e["accepted"]}',
            fill="black",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    buf = io.BytesIO()
    canvas.resize(
        (canvas.width * 3 // 4, canvas.height * 3 // 4),
        Image.Resampling.LANCZOS,
    ).save(buf, format="JPEG", quality=62, optimize=True, subsampling=2)
    path.with_suffix(".review.jpg.b64.txt").write_text(
        base64.b64encode(buf.getvalue()).decode("ascii") + "\n",
        encoding="ascii",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--st-membership", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--hand-model", type=Path, required=True)
    ap.add_argument("--lama-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    os.environ["LAMA_MODEL"] = str(args.lama_model)
    from simple_lama_inpainting import SimpleLama

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    st = read_csv(args.st_membership)
    target_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in target_rows]

    st_context = {r["sample_id"] for r in st if r["role"] == "context"}
    st_test = {r["sample_id"] for r in st if r["role"] == "test"}
    st_non_test = {r["sample_id"] for r in st if r["role"] != "test"}

    if len(target_ids) != 48 or any(sid not in st_context for sid in target_ids):
        raise ValueError("Targets must be the frozen 48 ST-context development rows")
    if set(target_ids) & st_test:
        raise ValueError("Primary-test target contamination")

    sources_by_video: dict[str, list[dict[str, str]]] = {}
    for sid in sorted(st_non_test):
        m = by_id[sid]
        sources_by_video.setdefault(m["video_id"], []).append(m)
    for rows in sources_by_video.values():
        rows.sort(key=lambda r: (int(r["frame_number"]), r["sample_id"]))

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    arm_model_path = hf_hub_download(MODEL_ID, ONNX_FILE, revision=MODEL_REVISION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = ort.InferenceSession(
        arm_model_path,
        opts,
        providers=["CPUExecutionProvider"],
    )
    hand_masker = TaskHandMasker(args.hand_model)
    lama = SimpleLama()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_root = args.output_dir / "final"
    mosaic_root = args.output_dir / "mosaic"
    qa_root = args.output_dir / "qa"

    records: list[dict] = []
    visuals: list[dict] = []

    try:
        with ZipFile(args.archive) as z:
            for index, sid in enumerate(target_ids, 1):
                meta = by_id[sid]
                target_rgb = decode_member(z, meta["archive_member"])
                target_mask, arm_px, hand_px, detected_hands = final_hand_arm_mask(
                    target_rgb, processor, session, hand_masker
                )
                _, target_kp, target_desc = target_features(target_rgb, target_mask)

                pool = [
                    r for r in sources_by_video.get(meta["video_id"], [])
                    if r["sample_id"] != sid
                ]
                pool.sort(
                    key=lambda r: (
                        abs(int(r["frame_number"]) - int(meta["frame_number"])),
                        r["sample_id"],
                    )
                )

                aligned = []
                for src_meta in pool:
                    if src_meta["sample_id"] in st_test:
                        raise RuntimeError("Forbidden ST test source reached alignment")
                    source_rgb = decode_member(z, src_meta["archive_member"])
                    source_mask, _, _, _ = final_hand_arm_mask(
                        source_rgb, processor, session, hand_masker
                    )
                    a = estimate_alignment(
                        target_rgb,
                        target_mask,
                        target_kp,
                        target_desc,
                        source_rgb,
                        source_mask,
                    )
                    if a is not None:
                        aligned.append({
                            "sample_id": src_meta["sample_id"],
                            "frame_distance": abs(
                                int(src_meta["frame_number"]) - int(meta["frame_number"])
                            ),
                            "alignment": a,
                        })

                mosaic, filled, accepted = best_source_mosaic(
                    target_rgb,
                    target_mask,
                    aligned,
                )
                mask_pixels = max(int((target_mask > 0).sum()), 1)
                real_fraction = float(((target_mask > 0) & filled).sum() / mask_pixels)
                residual = (target_mask > 0) & (~filled)
                residual_fraction = float(residual.sum() / mask_pixels)

                if np.any(residual):
                    final = run_lama_scaled(
                        lama,
                        mosaic,
                        residual.astype(np.uint8) * 255,
                    )
                else:
                    final = mosaic.copy()
                final[target_mask == 0] = target_rgb[target_mask == 0]

                for root, arr in ((mosaic_root, mosaic), (final_root, final)):
                    out = root / meta["archive_member"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(out, format="PNG")

                outside_changed = bool(
                    np.any(final[target_mask == 0] != target_rgb[target_mask == 0])
                )

                rec = {
                    "sample_id": sid,
                    "video_id": meta["video_id"],
                    "frame_number": int(meta["frame_number"]),
                    "label": meta["label"],
                    "mask_fraction": float((target_mask > 0).mean()),
                    "arm_pixels": arm_px,
                    "hand_pixels": hand_px,
                    "detected_hands": detected_hands,
                    "source_pool_size": len(pool),
                    "accepted_sources": len(accepted),
                    "real_pixel_fill_fraction": real_fraction,
                    "residual_fraction": residual_fraction,
                    "meets_fill_gate": bool(
                        real_fraction >= 0.80 and residual_fraction <= 0.20
                    ),
                    "outside_mask_changed": outside_changed,
                    "accepted_source_details": accepted,
                }
                records.append(rec)
                visuals.append({
                    "sample_id": sid,
                    "original": target_rgb,
                    "mask": target_mask,
                    "mosaic": mosaic,
                    "filled": filled,
                    "final": final,
                    "real_fraction": real_fraction,
                    "pool": len(pool),
                    "accepted": len(accepted),
                })
                print(
                    f"{index}/48 {sid} pool={len(pool)} accepted={len(accepted)} "
                    f"real={real_fraction:.3f} residual={residual_fraction:.3f}",
                    flush=True,
                )
    finally:
        hand_masker.close()

    for start in range(0, 48, 12):
        build_sheet(
            visuals[start:start+12],
            qa_root / f"broad_mosaic_{start//12+1}.jpg",
        )

    real = np.asarray([r["real_pixel_fill_fraction"] for r in records], dtype=float)
    residual = np.asarray([r["residual_fraction"] for r in records], dtype=float)
    pool_sizes = np.asarray([r["source_pool_size"] for r in records], dtype=int)
    accepted_counts = np.asarray([r["accepted_sources"] for r in records], dtype=int)
    gate = np.asarray([r["meets_fill_gate"] for r in records], dtype=bool)

    summary = {
        "status": "BROAD_TEMPORAL_MOSAIC_48_READY_FOR_VISUAL_REVIEW",
        "images": 48,
        "primary_test_images_used": 0,
        "source_population": "same-video ST non-test rows only",
        "classifier_predictions_used": False,
        "lama_model_sha256": sha256_file(args.lama_model),
        "source_pool_size": {
            "mean": float(pool_sizes.mean()),
            "median": float(np.median(pool_sizes)),
            "max": int(pool_sizes.max()),
        },
        "accepted_sources": {
            "mean": float(accepted_counts.mean()),
            "median": float(np.median(accepted_counts)),
            "max": int(accepted_counts.max()),
        },
        "real_pixel_fill_fraction": {
            "mean": float(real.mean()),
            "median": float(np.median(real)),
            "p10": float(np.quantile(real, 0.10)),
            "min": float(real.min()),
        },
        "residual_fraction": {
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "p90": float(np.quantile(residual, 0.90)),
            "max": float(residual.max()),
        },
        "images_meeting_fill_gate": int(gate.sum()),
        "fraction_images_meeting_fill_gate": float(gate.mean()),
        "quantitative_gate_pass": bool(gate.mean() >= 0.90),
        "outside_mask_changed_images": int(
            sum(r["outside_mask_changed"] for r in records)
        ),
        "visual_gate": "PENDING",
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
