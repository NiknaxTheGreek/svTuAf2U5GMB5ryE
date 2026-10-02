from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
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
    schp_arm_mask,
)

MAX_NEIGHBORS = 8
ALIGN_SCALE = 0.5
ORB_NFEATURES = 4000
LOWE_RATIO = 0.75
RANSAC_REPROJ = 4.0
MIN_GOOD = 20
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.35
MIN_OVERLAP = 0.40
MAX_MEDIAN_GRAY_ERROR = 35.0


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    data = z.read(member)
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode {member}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def final_hand_arm_mask(
    rgb: np.ndarray,
    processor,
    session,
    hand_masker: TaskHandMasker,
) -> tuple[np.ndarray, int, int, int]:
    pil = Image.fromarray(rgb, mode="RGB")
    h, w, _ = rgb.shape
    arm = schp_arm_mask(pil, processor, session, ARM_IDS)
    hand, detected_hands = hand_masker.mask(rgb)
    mask = np.maximum(arm, hand)
    rad = max(3, int(round(0.008 * min(h, w))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = cv2.dilate(mask, ker, iterations=1)
    return mask, int((arm > 0).sum()), int((hand > 0).sum()), detected_hands


def scaled_homography_to_full(Hs: np.ndarray, scale: float) -> np.ndarray:
    S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.linalg.inv(S) @ Hs @ S


def target_features(rgb: np.ndarray, mask: np.ndarray):
    small = cv2.resize(rgb, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    valid = cv2.resize((mask == 0).astype(np.uint8) * 255, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=7)
    kp, desc = orb.detectAndCompute(gray, valid)
    return gray, kp, desc


def estimate_alignment(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    target_kp,
    target_desc,
    source_rgb: np.ndarray,
    source_mask: np.ndarray,
):
    source_small = cv2.resize(source_rgb, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA)
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_RGB2GRAY)
    source_valid = cv2.resize((source_mask == 0).astype(np.uint8) * 255, (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=7)
    source_kp, source_desc = orb.detectAndCompute(source_gray, source_valid)
    if target_desc is None or source_desc is None or len(target_kp) < MIN_GOOD or len(source_kp) < MIN_GOOD:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_GOOD:
        return None

    src_pts = np.float32([source_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hs, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ)
    if Hs is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / max(len(good), 1)
    if inliers < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
        return None

    H = scaled_homography_to_full(Hs, ALIGN_SCALE)
    h, w = target_rgb.shape[:2]
    warped = cv2.warpPerspective(
        source_rgb, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    warped_source_unmasked = cv2.warpPerspective(
        (source_mask == 0).astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    warped_valid = cv2.warpPerspective(
        np.ones(source_mask.shape, dtype=np.uint8), H, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)

    compare = (target_mask == 0) & warped_source_unmasked & warped_valid
    denom = max(int((target_mask == 0).sum()), 1)
    overlap_fraction = float(compare.sum() / denom)
    if overlap_fraction < MIN_OVERLAP:
        return None

    target_gray_full = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    errors = np.abs(target_gray_full.astype(np.int16) - warped_gray.astype(np.int16))[compare]
    median_error = float(np.median(errors)) if errors.size else 999.0
    if median_error > MAX_MEDIAN_GRAY_ERROR:
        return None

    quality = float((inliers * ratio) / (1.0 + median_error))
    fill_valid = (target_mask > 0) & warped_source_unmasked & warped_valid
    return {
        "warped": warped,
        "fill_valid": fill_valid,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": ratio,
        "overlap_fraction": overlap_fraction,
        "median_gray_abs_error": median_error,
        "quality": quality,
    }


def reconstruct(
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    candidates: list[dict],
):
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
            "good_matches": a["good_matches"],
            "inliers": a["inliers"],
            "inlier_ratio": a["inlier_ratio"],
            "overlap_fraction": a["overlap_fraction"],
            "median_gray_abs_error": a["median_gray_abs_error"],
            "quality": a["quality"],
            "weighted_score": score,
            "fill_pixels": int(valid.sum()),
        })

    real_fill = (target_mask > 0) & (weights > 0)
    mask_pixels = max(int((target_mask > 0).sum()), 1)
    real_fill_fraction = float(real_fill.sum() / mask_pixels)

    recon = target_rgb.copy().astype(np.float64)
    recon[real_fill] = accum[real_fill] / weights[real_fill, None]
    recon = np.clip(recon, 0, 255).astype(np.uint8)

    residual = (target_mask > 0) & (~real_fill)
    residual_fraction = float(residual.sum() / mask_pixels)

    # TELEA is restricted to residual holes after transfer from real neighboring pixels.
    if np.any(residual):
        bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, residual.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        recon = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Feather only the final target-mask seam.
    sigma = max(2.0, 0.0025 * min(h, w))
    alpha = cv2.GaussianBlur((target_mask > 0).astype(np.float32), (0, 0), sigma)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    out = target_rgb.astype(np.float32) * (1.0 - alpha) + recon.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, real_fill_fraction, residual_fraction, accepted


def make_mask_preview(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    overlay = rgb.copy().astype(np.float32)
    tint = np.zeros_like(overlay)
    tint[..., 0] = 255
    alpha = (mask > 0)[..., None].astype(np.float32) * 0.45
    overlay = overlay * (1 - alpha) + tint * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def build_sheet(entries, path: Path):
    cell = (180, 320)
    gap = 8
    label_h = 44
    rows = len(entries)
    canvas = Image.new("RGB", (3 * cell[0] + 4 * gap, rows * (cell[1] + label_h + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for i, e in enumerate(entries):
        y = gap + i * (cell[1] + label_h + gap)
        images = [
            Image.fromarray(e["original"]),
            make_mask_preview(e["original"], e["mask"]),
            Image.fromarray(e["reconstructed"]),
        ]
        labels = ["BEFORE", "MASK", f'VIDEO real={e["real_fill_fraction"]:.2f}']
        for col, (im, lab) in enumerate(zip(images, labels)):
            thumb = ImageOps.fit(im.convert("RGB"), cell, method=Image.Resampling.BILINEAR)
            x = gap + col * (cell[0] + gap)
            canvas.paste(thumb, (x, y))
            draw.text((x, y + cell[1] + 4), lab, fill="black")
        draw.text(
            (gap, y + cell[1] + 20),
            f'{e["sample_id"]} src={e["accepted_count"]} residual={e["residual_fraction"]:.2f}',
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)
    buf = io.BytesIO()
    canvas.resize((canvas.width * 3 // 4, canvas.height * 3 // 4), Image.Resampling.LANCZOS).save(
        buf, format="JPEG", quality=62, optimize=True, subsampling=2
    )
    path.with_suffix(".review.jpg.b64.txt").write_text(base64.b64encode(buf.getvalue()).decode("ascii") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--st-membership", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--hand-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    st = read_csv(args.st_membership)
    st_context_ids = {r["sample_id"] for r in st if r["role"] == "context"}
    st_test_ids = {r["sample_id"] for r in st if r["role"] == "test"}
    sample_rows = read_csv(args.sample)
    target_ids = [r["sample_id"] for r in sample_rows]
    if len(target_ids) != 48 or any(sid not in st_context_ids for sid in target_ids):
        raise ValueError("Targets are not the exact 48 ST-context development images")
    if set(target_ids) & st_test_ids:
        raise ValueError("Primary-test target contamination")

    context_by_video = {}
    for sid in sorted(st_context_ids):
        m = by_id[sid]
        context_by_video.setdefault(m["video_id"], []).append(m)
    for rows in context_by_video.values():
        rows.sort(key=lambda r: (int(r["frame_number"]), r["sample_id"]))

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    model_path = hf_hub_download(MODEL_ID, ONNX_FILE, revision=MODEL_REVISION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(8, __import__("os").cpu_count() or 1))
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_root = args.output_dir / "reconstructed"
    mask_root = args.output_dir / "masks"
    qa_root = args.output_dir / "qa"
    records = []
    visual_entries = []

    with ZipFile(args.archive) as z, TaskHandMasker(args.hand_model) as hand_masker:
        for index, sid in enumerate(target_ids, 1):
            target_meta = by_id[sid]
            target_rgb = decode_member(z, target_meta["archive_member"])
            target_mask, arm_px, hand_px, detected_hands = final_hand_arm_mask(
                target_rgb, processor, session, hand_masker
            )
            _, target_kp, target_desc = target_features(target_rgb, target_mask)

            pool = [
                r for r in context_by_video[target_meta["video_id"]]
                if r["sample_id"] != sid
            ]
            pool.sort(key=lambda r: (abs(int(r["frame_number"]) - int(target_meta["frame_number"])), r["sample_id"]))
            pool = pool[:MAX_NEIGHBORS]

            aligned = []
            attempts = []
            for src_meta in pool:
                source_rgb = decode_member(z, src_meta["archive_member"])
                source_mask, _, _, _ = final_hand_arm_mask(
                    source_rgb, processor, session, hand_masker
                )
                a = estimate_alignment(
                    target_rgb, target_mask, target_kp, target_desc, source_rgb, source_mask
                )
                attempt = {
                    "sample_id": src_meta["sample_id"],
                    "frame_distance": abs(int(src_meta["frame_number"]) - int(target_meta["frame_number"])),
                    "accepted": a is not None,
                }
                attempts.append(attempt)
                if a is not None:
                    aligned.append({**attempt, "alignment": a})

            reconstructed, real_fill_fraction, residual_fraction, accepted = reconstruct(
                target_rgb, target_mask, aligned
            )

            out_path = clean_root / target_meta["archive_member"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstructed).save(out_path, quality=95)

            mask_path = mask_root / (target_meta["archive_member"] + ".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target_mask).save(mask_path)

            rec = {
                "sample_id": sid,
                "video_id": target_meta["video_id"],
                "frame_number": int(target_meta["frame_number"]),
                "label": target_meta["label"],
                "mask_fraction": float((target_mask > 0).mean()),
                "arm_pixels": arm_px,
                "hand_pixels": hand_px,
                "detected_hands": detected_hands,
                "source_candidates": len(pool),
                "accepted_sources": len(accepted),
                "real_pixel_fill_fraction": real_fill_fraction,
                "residual_fraction": residual_fraction,
                "eligible_by_fill_gate": bool(real_fill_fraction >= 0.80 and residual_fraction <= 0.20),
                "accepted_source_details": accepted,
                "attempted_sources": attempts,
            }
            records.append(rec)
            visual_entries.append({
                "sample_id": sid,
                "original": target_rgb,
                "mask": target_mask,
                "reconstructed": reconstructed,
                "real_fill_fraction": real_fill_fraction,
                "residual_fraction": residual_fraction,
                "accepted_count": len(accepted),
            })
            print(f"{index}/48 {sid} real={real_fill_fraction:.3f} residual={residual_fraction:.3f} accepted={len(accepted)}", flush=True)

    for start in range(0, 48, 12):
        build_sheet(visual_entries[start:start+12], qa_root / f"video_reconstruction_{start//12+1}.jpg")

    eligible = [r["eligible_by_fill_gate"] for r in records]
    real = np.asarray([r["real_pixel_fill_fraction"] for r in records], dtype=float)
    residual = np.asarray([r["residual_fraction"] for r in records], dtype=float)
    accepted_counts = np.asarray([r["accepted_sources"] for r in records], dtype=int)

    summary = {
        "status": "VIDEO_RECONSTRUCTION_48_READY_FOR_VISUAL_REVIEW",
        "images": 48,
        "primary_test_images_used": 0,
        "source_population": "ST context-only rows from same canonical video",
        "classifier_predictions_used": False,
        "images_with_accepted_source": int(np.sum(accepted_counts > 0)),
        "images_meeting_fill_gate": int(np.sum(eligible)),
        "fraction_images_meeting_fill_gate": float(np.mean(eligible)),
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
        "accepted_sources": {
            "mean": float(accepted_counts.mean()),
            "median": float(np.median(accepted_counts)),
            "max": int(accepted_counts.max()),
        },
        "quantitative_gate_pass": bool(float(np.mean(eligible)) >= 0.90),
        "visual_gate": "PENDING",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
