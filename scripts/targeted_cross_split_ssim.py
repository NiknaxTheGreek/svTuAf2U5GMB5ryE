#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

SIZE = (256, 455)  # preserve ~1080x1920 aspect ratio
VERY_D = 4
VERY_P = 4

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize(SIZE)
        return np.asarray(im, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    imgs = []
    dh = []
    ph = []
    seq = []
    split = []
    label = []

    for r in rows:
        p = args.image_root / r["path"]
        if not p.exists():
            p = args.image_root / "images" / r["path"]
        arr = load_gray(p)
        imgs.append(arr)
        with Image.open(p) as im:
            im = im.convert("RGB")
            dh.append(int(str(imagehash.dhash(im, hash_size=8)), 16))
            ph.append(int(str(imagehash.phash(im, hash_size=8)), 16))
        seq.append(f'{r["label"]}__{r["video_id"]}')
        split.append(r["split"])
        label.append(r["label"])

    candidates = []
    n = len(rows)
    for i in range(n-1):
        for j in range(i+1, n):
            if split[i] == split[j]:
                continue
            d = (dh[i] ^ dh[j]).bit_count()
            if d > VERY_D:
                continue
            p = (ph[i] ^ ph[j]).bit_count()
            if p > VERY_P:
                continue
            score = float(ssim(imgs[i], imgs[j], data_range=255))
            if seq[i] == seq[j]:
                bucket = "same_sequence_cross_split"
            elif label[i] == label[j]:
                bucket = "different_sequence_cross_split_same_label"
            else:
                bucket = "different_sequence_cross_split_cross_label"
            candidates.append({
                "bucket": bucket,
                "ssim": score,
                "dhash": d,
                "phash": p,
                "path_a": rows[i]["path"],
                "path_b": rows[j]["path"],
                "sequence_a": seq[i],
                "sequence_b": seq[j],
                "label_a": label[i],
                "label_b": label[j],
            })

    candidates.sort(key=lambda x: x["ssim"], reverse=True)

    with (args.out_dir / "cross_split_very_high_similarity_ssim.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)

    per_bucket = {}
    for bucket in sorted(set(x["bucket"] for x in candidates)):
        vals = np.array([x["ssim"] for x in candidates if x["bucket"] == bucket], dtype=float)
        per_bucket[bucket] = {
            "pair_count": int(len(vals)),
            "mean_ssim": float(np.mean(vals)),
            "median_ssim": float(np.median(vals)),
            "q1_ssim": float(np.quantile(vals, 0.25)),
            "q3_ssim": float(np.quantile(vals, 0.75)),
            "ssim_ge_0_95": int(np.sum(vals >= 0.95)),
            "ssim_ge_0_98": int(np.sum(vals >= 0.98)),
            "ssim_ge_0_99": int(np.sum(vals >= 0.99)),
        }

    vals = np.array([x["ssim"] for x in candidates], dtype=float)
    summary = {
        "image_count": n,
        "screen_rule": "cross-split only; dHash <= 4 AND pHash <= 4",
        "ssim_resolution": f"{SIZE[0]}x{SIZE[1]} grayscale",
        "candidate_pair_count": int(len(candidates)),
        "overall": {
            "mean_ssim": float(np.mean(vals)),
            "median_ssim": float(np.median(vals)),
            "q1_ssim": float(np.quantile(vals, 0.25)),
            "q3_ssim": float(np.quantile(vals, 0.75)),
            "ssim_ge_0_95": int(np.sum(vals >= 0.95)),
            "ssim_ge_0_98": int(np.sum(vals >= 0.98)),
            "ssim_ge_0_99": int(np.sum(vals >= 0.99)),
        },
        "per_bucket": per_bucket,
        "top_10_pairs": candidates[:10],
        "scope_note": "SSIM confirms structural visual similarity after perceptual-hash screening. It does not establish identical semantics or label correctness. No images are removed automatically."
    }

    (args.out_dir / "cross_split_ssim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
