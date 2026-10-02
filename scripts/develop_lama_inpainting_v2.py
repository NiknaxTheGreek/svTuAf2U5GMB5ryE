from __future__ import annotations

import argparse
import base64
import csv
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
    ARM_IDS,
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

MAX_SIDE = 1024
MAX_NEIGHBORS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside_only_feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    binary = (mask > 0).astype(np.float32)
    blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
    return np.where(binary > 0, blurred, 0.0).astype(np.float32)


def run_lama_scaled(lama, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return rgb.copy()

    h, w = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        sw = max(8, int(round(w * scale)))
        sh = max(8, int(round(h * scale)))
        image_small = Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS)
        mask_small = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
            (sw, sh), Image.Resampling.NEAREST
        )
    else:
        image_small = Image.fromarray(rgb)
        mask_small = Image.fromarray((mask > 0).astype(np.uint8) * 255)

    generated = lama(image_small, mask_small).convert("RGB")
    if generated.size != (w, h):
        generated = generated.resize((w, h), Image.Resampling.LANCZOS)
    gen = np.asarray(generated, dtype=np.uint8)

    # Preserve all pixels outside the requested mask exactly.
    alpha = inside_only_feather(mask, max(1.5, 0.002 * min(h, w)))[..., None]
    out = rgb.astype(np.float32) * (1.0 - alpha) + gen.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[mask == 0] = rgb[mask == 0]
    return out


def video_prefill(target_rgb, target_mask, candidates):
    h, w = target_mask.shape
    accum = np.zeros((h, w, 3), dtype=np.float64)
    weights = np.zeros((h, w), dtype=np.float64)
    accepted = []

    for cand in candidates:
        a = cand["alignment"]
        frame_distance = cand["frame_distance"]
        score = a["quality"] / (1.0 + 0.05 * frame_distance)
        valid = a["fill_valid"]
        if not np.any(valid):
            continue
        accum[valid] += a["warped"][valid].astype(np.float64) * score
        weights[valid] += score
        accepted.append({
            "sample_id": cand["sample_id"],
            "frame_distance": frame_distance,
            "inliers": a["inliers"],
            "inlier_ratio": a["inlier_ratio"],
            "overlap_fraction": a["overlap_fraction"],
            "median_gray_abs_error": a["median_gray_abs_error"],
            "weighted_score": score,
            "fill_pixels": int(valid.sum()),
        })

    real_fill = (target_mask > 0) & (weights > 0)
    out = target_rgb.copy().astype(np.float64)
    out[real_fill] = accum[real_fill] / weights[real_fill, None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    mask_pixels = max(int((target_mask > 0).sum()), 1)
    fraction = float(real_fill.sum() / mask_pixels)
    return out, real_fill, fraction, accepted


def build_sheet(entries, path: Path):
    cell = (150, 267)
    gap = 7
    label_h = 48
    rows = len(entries)
    width = 4 * cell[0] + 5 * gap
    height = rows * (cell[1] + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for i, e in enumerate(entries):
        y = gap + i * (cell[1] + label_h + gap)
        mask_overlay = e["original"].copy().astype(np.float32)
        tint = np.zeros_like(mask_overlay)
        tint[..., 0] = 255
        a = (e["mask"] > 0)[..., None].astype(np.float32) * 0.45
        mask_overlay = np.clip(mask_overlay * (1 - a) + tint * a, 0, 255).astype(np.uint8)

        imgs = [
            e["original"],
            mask_overlay,
            e["lama_full"],
            e["video_lama"],
        ]
        labels = [
            "BEFORE",
            "MASK",
            "LAMA FULL",
            f'VIDEO+LAMA real={e["real_fill_fraction"]:.2f}',
        ]
        for col, (arr, label) in enumerate(zip(imgs, labels)):
            thumb = ImageOps.fit(
                Image.fromarray(arr).convert("RGB"),
                cell,
                method=Image.Resampling.BILINEAR,
            )
            x = gap + col * (cell[0] + gap)
            canvas.paste(thumb, (x, y))
            draw.text((x, y + cell[1] + 3), label, fill="black")

        draw.text(
            (gap, y + cell[1] + 22),
            f'{e["sample_id"]} accepted={e["accepted_count"]}',
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


def main():
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
    context_ids = {r["sample_id"] for r in st if r["role"] == "context"}
    test_ids = {r["sample_id"] for r in st if r["role"] == "test"}
    sample_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in sample_rows]

    if len(target_ids) != 48 or any(sid not in context_ids for sid in target_ids):
        raise ValueError("Targets are not the frozen 48 ST-context development images")
    if set(target_ids) & test_ids:
        raise ValueError("Primary-test contamination")

    context_by_video = {}
    for sid in sorted(context_ids):
        m = by_id[sid]
        context_by_video.setdefault(m["video_id"], []).append(m)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    arm_model_path = hf_hub_download(MODEL_ID, ONNX_FILE, revision=MODEL_REVISION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = ort.InferenceSession(
        arm_model_path, opts, providers=["CPUExecutionProvider"]
    )
    hand_masker = TaskHandMasker(args.hand_model)
    lama = SimpleLama()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_root = args.output_dir / "lama_full"
    hybrid_root = args.output_dir / "video_lama"
    qa_root = args.output_dir / "qa"
    records = []
    visual_entries = []

    try:
        with ZipFile(args.archive) as z:
            for index, sid in enumerate(target_ids, 1):
                meta = by_id[sid]
                target_rgb = decode_member(z, meta["archive_member"])
                target_mask, arm_px, hand_px, detected_hands = final_hand_arm_mask(
                    target_rgb, processor, session, hand_masker
                )

                # Variant 1: LaMa over the full frozen hand/arm mask.
                lama_full = run_lama_scaled(lama, target_rgb, target_mask)

                # Build the same context-only real-pixel transfer used in Stage 2.
                _, target_kp, target_desc = target_features(target_rgb, target_mask)
                pool = [
                    r for r in context_by_video.get(meta["video_id"], [])
                    if r["sample_id"] != sid
                ]
                pool.sort(
                    key=lambda r: (
                        abs(int(r["frame_number"]) - int(meta["frame_number"])),
                        r["sample_id"],
                    )
                )
                pool = pool[:MAX_NEIGHBORS]

                aligned = []
                for src_meta in pool:
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

                prefilled, real_fill, real_fraction, accepted = video_prefill(
                    target_rgb, target_mask, aligned
                )
                residual = (target_mask > 0) & (~real_fill)
                if np.any(residual):
                    hybrid = run_lama_scaled(
                        lama, prefilled, residual.astype(np.uint8) * 255
                    )
                else:
                    hybrid = prefilled
                hybrid[target_mask == 0] = target_rgb[target_mask == 0]

                for root, arr in [(full_root, lama_full), (hybrid_root, hybrid)]:
                    out = root / meta["archive_member"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(out, format="PNG")

                outside = target_mask == 0
                full_outside_changed = int(np.any(lama_full[outside] != target_rgb[outside]))
                hybrid_outside_changed = int(np.any(hybrid[outside] != target_rgb[outside]))

                rec = {
                    "sample_id": sid,
                    "video_id": meta["video_id"],
                    "frame_number": int(meta["frame_number"]),
                    "label": meta["label"],
                    "mask_fraction": float((target_mask > 0).mean()),
                    "arm_pixels": arm_px,
                    "hand_pixels": hand_px,
                    "detected_hands": detected_hands,
                    "accepted_video_sources": len(accepted),
                    "real_pixel_fill_fraction": real_fraction,
                    "lama_full_outside_mask_changed": full_outside_changed,
                    "video_lama_outside_mask_changed": hybrid_outside_changed,
                    "accepted_source_details": accepted,
                }
                records.append(rec)
                visual_entries.append({
                    "sample_id": sid,
                    "original": target_rgb,
                    "mask": target_mask,
                    "lama_full": lama_full,
                    "video_lama": hybrid,
                    "real_fill_fraction": real_fraction,
                    "accepted_count": len(accepted),
                })

                print(
                    f"{index}/48 {sid} mask={rec['mask_fraction']:.3f} "
                    f"real={real_fraction:.3f} accepted={len(accepted)}",
                    flush=True,
                )
    finally:
        hand_masker.close()

    for start in range(0, 48, 12):
        build_sheet(
            visual_entries[start:start+12],
            qa_root / f"lama_comparison_{start//12+1}.jpg",
        )

    summary = {
        "status": "LAMA_48_READY_FOR_VISUAL_REVIEW",
        "images": 48,
        "primary_test_images_used": 0,
        "classifier_predictions_used": False,
        "package": "simple-lama-inpainting==0.1.2",
        "lama_model_sha256": sha256_file(args.lama_model),
        "lama_max_side": MAX_SIDE,
        "variants": ["lama_full", "video_lama_residual"],
        "outside_mask_preservation": {
            "lama_full_images_changed_outside_mask": int(
                sum(r["lama_full_outside_mask_changed"] for r in records)
            ),
            "video_lama_images_changed_outside_mask": int(
                sum(r["video_lama_outside_mask_changed"] for r in records)
            ),
        },
        "video_real_pixel_fill_fraction": {
            "mean": float(np.mean([r["real_pixel_fill_fraction"] for r in records])),
            "median": float(np.median([r["real_pixel_fill_fraction"] for r in records])),
        },
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
