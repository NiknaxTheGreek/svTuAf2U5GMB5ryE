#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = (64, 64)
BORDER_FRAC = 0.18

def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(SIZE), dtype=np.float32) / 255.0

def clip_signature(frames: list[np.ndarray]) -> dict[str, np.ndarray]:
    stack = np.stack(frames, axis=0)
    mean_img = np.mean(stack, axis=0)
    median_img = np.median(stack, axis=0)

    h, w, _ = mean_img.shape
    bh = max(1, int(round(h * BORDER_FRAC)))
    bw = max(1, int(round(w * BORDER_FRAC)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True

    border = mean_img[mask]
    center = mean_img[~mask]

    # Global color distribution: 16 bins x 3 channels.
    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(stack[..., c].ravel(), bins=16, range=(0, 1), density=True)
        hist_parts.append(hist.astype(np.float32))
    color_hist = np.concatenate(hist_parts)
    color_hist /= (np.linalg.norm(color_hist) + 1e-12)

    return {
        "mean_full": mean_img.reshape(-1),
        "median_full": median_img.reshape(-1),
        "mean_border": border.reshape(-1),
        "mean_center": center.reshape(-1),
        "color_hist": color_hist,
    }

def l1(a, b):
    return float(np.mean(np.abs(a - b)))

def cosine_distance(a, b):
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / den)

def rank_of(value, arr):
    # 1 = closest. Average rank among ties.
    arr = np.asarray(arr)
    return float(1 + np.sum(arr < value) + 0.5 * np.sum(arr == value) - 0.5)

def pct_rank(rank, n):
    if n <= 1:
        return 0.0
    return float((rank - 1) / (n - 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["video_id"])].append(r)

    signatures = {}
    clip_meta = {}
    for key, rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda x: int(x["frame_num"]))
        imgs = []
        for r in rr:
            p = args.image_root / r["path"]
            if not p.exists():
                p = args.image_root / "images" / r["path"]
            imgs.append(load_rgb(p))
        signatures[key] = clip_signature(imgs)
        clip_meta[key] = {
            "frames": len(rr),
            "min_frame": min(int(r["frame_num"]) for r in rr),
            "max_frame": max(int(r["frame_num"]) for r in rr),
        }

    flip_ids = sorted(vid for (label, vid) in signatures if label == "flip")
    notflip_ids = sorted(vid for (label, vid) in signatures if label == "notflip")
    paired_ids = sorted(set(flip_ids) & set(notflip_ids))

    metrics = [
        ("mean_full_l1", "mean_full", l1),
        ("median_full_l1", "median_full", l1),
        ("border_l1", "mean_border", l1),
        ("center_l1", "mean_center", l1),
        ("color_hist_cosine", "color_hist", cosine_distance),
    ]

    # Build cross-label distance matrices by metric.
    dists = {}
    for mname, skey, fn in metrics:
        mat = np.zeros((len(notflip_ids), len(flip_ids)), dtype=np.float32)
        for i, nid in enumerate(notflip_ids):
            a = signatures[("notflip", nid)][skey]
            for j, fid in enumerate(flip_ids):
                b = signatures[("flip", fid)][skey]
                mat[i, j] = fn(a, b)
        dists[mname] = mat

    nidx = {vid:i for i,vid in enumerate(notflip_ids)}
    fidx = {vid:j for j,vid in enumerate(flip_ids)}

    rows_out = []
    for vid in paired_ids:
        rec = {
            "video_id": vid,
            "notflip_frames": clip_meta[("notflip", vid)]["frames"],
            "flip_frames": clip_meta[("flip", vid)]["frames"],
        }
        pct_parts = []
        for mname, _, _ in metrics:
            mat = dists[mname]
            i, j = nidx[vid], fidx[vid]
            same = float(mat[i, j])

            # notflip -> all flips
            r_nf = rank_of(same, mat[i, :])
            p_nf = pct_rank(r_nf, len(flip_ids))

            # flip -> all notflips
            r_f = rank_of(same, mat[:, j])
            p_f = pct_rank(r_f, len(notflip_ids))

            rec[f"{mname}_distance"] = same
            rec[f"{mname}_rank_notflip_to_flip"] = r_nf
            rec[f"{mname}_pct_notflip_to_flip"] = p_nf
            rec[f"{mname}_rank_flip_to_notflip"] = r_f
            rec[f"{mname}_pct_flip_to_notflip"] = p_f
            pct_parts.extend([p_nf, p_f])

        rec["composite_mean_percentile"] = float(np.mean(pct_parts))
        rec["composite_median_percentile"] = float(np.median(pct_parts))
        rows_out.append(rec)

    with (args.out_dir / "same_id_session_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    # Aggregate nearest-neighbor evidence.
    agg = {
        "paired_ids": len(paired_ids),
        "notflip_clip_count": len(notflip_ids),
        "flip_clip_count": len(flip_ids),
        "cross_label_pair_count": len(notflip_ids) * len(flip_ids),
        "signature_metrics": [m[0] for m in metrics],
        "interpretation": "For each clip, lower same-ID rank/percentile means its opposite-label same-ID clip is visually closer than unrelated IDs. Rank 1 means nearest opposite-label clip.",
        "per_metric": {},
        "composite": {
            "same_id_top_1_count_by_composite_mean_percentile": int(sum(r["composite_mean_percentile"] <= 0.01 for r in rows_out)),
            "same_id_top_5pct_count": int(sum(r["composite_mean_percentile"] <= 0.05 for r in rows_out)),
            "same_id_top_10pct_count": int(sum(r["composite_mean_percentile"] <= 0.10 for r in rows_out)),
            "median_composite_mean_percentile": float(np.median([r["composite_mean_percentile"] for r in rows_out])),
            "mean_composite_mean_percentile": float(np.mean([r["composite_mean_percentile"] for r in rows_out])),
        },
        "scope_note": "Whole-clip appearance similarity can support a shared recording/session hypothesis, but cannot prove the clips came from the same uninterrupted recording. Similar books, person, table, camera, or lighting can also create closeness."
    }

    for mname, _, _ in metrics:
        nf_ranks = np.array([r[f"{mname}_rank_notflip_to_flip"] for r in rows_out])
        f_ranks = np.array([r[f"{mname}_rank_flip_to_notflip"] for r in rows_out])
        nf_pct = np.array([r[f"{mname}_pct_notflip_to_flip"] for r in rows_out])
        f_pct = np.array([r[f"{mname}_pct_flip_to_notflip"] for r in rows_out])
        agg["per_metric"][mname] = {
            "notflip_to_flip_rank1_count": int(np.sum(nf_ranks == 1)),
            "flip_to_notflip_rank1_count": int(np.sum(f_ranks == 1)),
            "notflip_to_flip_top5_count": int(np.sum(nf_ranks <= 5)),
            "flip_to_notflip_top5_count": int(np.sum(f_ranks <= 5)),
            "median_pct_notflip_to_flip": float(np.median(nf_pct)),
            "median_pct_flip_to_notflip": float(np.median(f_pct)),
        }

    (args.out_dir / "same_id_session_similarity_summary.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
