#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import mediapipe as mp

MODEL_ID = "pirocheto/schp-pascal-7"
MODEL_REVISION = "e97480b846bf0f23a9f9b7ab673dc1c86af89467"
ONNX_FILE = "onnx/schp-pascal-7-int8-static.onnx"
ARM_IDS = [3, 4]


class TaskHandMasker:
    """Compatibility replacement for removed mp.solutions.hands API.

    Preserves historical semantics: static-image detection, up to two hands,
    convex-hull mask, and 2.5%-of-short-side dilation. The only compatibility
    change is MediaPipe Tasks HandLandmarker with the repository-verified model.
    """
    def __init__(self, model_path: Path):
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def close(self):
        self.landmarker.close()

    def mask(self, rgb: np.ndarray) -> tuple[np.ndarray, int]:
        h, w, _ = rgb.shape
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb, dtype=np.uint8),
        )
        result = self.landmarker.detect(img)
        mask = np.zeros((h, w), np.uint8)
        for hand in result.hand_landmarks:
            pts = np.array(
                [
                    [
                        int(round(float(lm.x) * (w - 1))),
                        int(round(float(lm.y) * (h - 1))),
                    ]
                    for lm in hand
                ],
                dtype=np.int32,
            )
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            radius = max(10, int(round(0.025 * min(h, w))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask, len(result.hand_landmarks)


def schp_arm_mask(pil_img, processor, session, arm_ids):
    orig_w, orig_h = pil_img.size
    inputs = processor(images=pil_img, return_tensors="np")
    pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    target = "logits" if "logits" in out_names else out_names[0]
    logits = session.run([target], {input_name: pixel_values})[0]
    pred = logits.argmax(axis=1)[0].astype(np.uint8)
    mask512 = np.isin(pred, np.asarray(arm_ids)).astype(np.uint8) * 255
    return cv2.resize(mask512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def clean_one(src: Path, dst: Path, processor, session, hand_masker, arm_ids):
    pil = Image.open(src).convert("RGB")
    rgb = np.asarray(pil)
    h, w, _ = rgb.shape

    arm = schp_arm_mask(pil, processor, session, arm_ids)
    hand, detected_hands = hand_masker.mask(rgb)
    mask = np.maximum(arm, hand)

    # Exact historical post-union morphology.
    rad = max(3, int(round(0.008 * min(h, w))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = cv2.dilate(mask, ker, iterations=1)

    frac = float((mask > 0).mean())

    # Exact historical half-resolution TELEA recipe.
    scale = 0.5
    sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    cleaned_small = cv2.inpaint(small, msmall, 5, cv2.INPAINT_TELEA)
    cleaned_up = cv2.resize(cleaned_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Exact historical feathered compositing.
    feather = max(3, int(round(0.004 * min(h, w))))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
    alpha = np.clip(alpha[..., None], 0, 1)
    out = bgr.astype(np.float32) * (1 - alpha) + cleaned_up.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return frac, int((arm > 0).sum()), int((hand > 0).sum()), detected_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--hand-model", type=Path, required=True)
    args = ap.parse_args()

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = sorted(rows, key=lambda r: (r["label"], r["video_id"], int(r["frame_num"]), r["split"]))

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    model_path = hf_hub_download(MODEL_ID, ONNX_FILE, revision=MODEL_REVISION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    report = []
    hand_masker = TaskHandMasker(args.hand_model)
    try:
        for k, r in enumerate(rows, 1):
            src = args.image_root / r["path"]
            if not src.exists():
                src = args.image_root / "images" / r["path"]
            dst = args.out_root / r["path"]
            frac, arm_px, hand_px, detected_hands = clean_one(
                src, dst, processor, session, hand_masker, ARM_IDS
            )
            report.append({
                "path": r["path"],
                "label": r["label"],
                "video_id": r["video_id"],
                "frame_num": r["frame_num"],
                "split": r["split"],
                "mask_fraction": frac,
                "arm_pixels": arm_px,
                "hand_pixels": hand_px,
                "detected_hands": detected_hands,
                "cleaned_path": str(dst),
            })
            if k % 25 == 0 or k == len(rows):
                print(f"{k}/{len(rows)}", flush=True)
    finally:
        hand_masker.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    vals = np.asarray([x["mask_fraction"] for x in report], dtype=float)
    summary = {
        "processed": len(report),
        "arm_model": MODEL_ID,
        "arm_model_revision": MODEL_REVISION,
        "arm_onnx_file": ONNX_FILE,
        "arm_label_ids": ARM_IDS,
        "hand_detector": "MediaPipe Tasks HandLandmarker compatibility replacement",
        "hand_detection_confidence": 0.35,
        "hand_presence_confidence": 0.35,
        "num_hands": 2,
        "images_with_hand_detection": int(sum(x["detected_hands"] > 0 for x in report)),
        "images_without_hand_detection": int(sum(x["detected_hands"] == 0 for x in report)),
        "mean_mask_fraction": float(vals.mean()),
        "median_mask_fraction": float(np.median(vals)),
        "zero_mask_images": int(sum(v == 0 for v in vals)),
        "method": "Pinned SCHP Pascal-7 ONNX arm segmentation + verified MediaPipe Tasks hand hulls; historical union morphology; half-resolution TELEA radius 5; cubic upsample; feathered compositing.",
        "compatibility_note": "Legacy mp.solutions.hands was unavailable in current MediaPipe. Only that API/model invocation was replaced; downstream hand-hull geometry, dilation, arm segmentation, union morphology, TELEA and feathering retain historical semantics.",
        "note": "Derived ablation dataset only. Canonical originals are never modified."
    }
    (args.report.parent / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
