#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20260925
N_SHUFFLES = 200
RESIZE = (64, 64)

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize(RESIZE), dtype=np.float32) / 255.0
    return arr

def dhash(arr: np.ndarray) -> np.ndarray:
    # 64x64 input -> reduce to 9x8 for 64-bit dHash
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize((9, 8))
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)

def pair_dist(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(a - b)))
    ha = dhash(a)
    hb = dhash(b)
    ham = float(np.count_nonzero(ha != hb))
    return mae, ham

def build_pairwise(arrays):
    n = len(arrays)
    mae = np.zeros((n, n), dtype=np.float32)
    ham = np.zeros((n, n), dtype=np.float32)
    hashes = [dhash(a) for a in arrays]
    for i in range(n):
        for j in range(i + 1, n):
            m = float(np.mean(np.abs(arrays[i] - arrays[j])))
            h = float(np.count_nonzero(hashes[i] != hashes[j]))
            mae[i, j] = mae[j, i] = m
            ham[i, j] = ham[j, i] = h
    return mae, ham

def mean_adjacent_distance(order, mae_matrix, ham_matrix):
    if len(order) < 2:
        return (math.nan, math.nan)
    idx_a = np.asarray(order[:-1], dtype=int)
    idx_b = np.asarray(order[1:], dtype=int)
    return (
        float(np.mean(mae_matrix[idx_a, idx_b])),
        float(np.mean(ham_matrix[idx_a, idx_b])),
    )

def zscore(obs, vals):
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (obs - mu) / sd if sd > 0 else 0.0, mu, sd

def percentile_lower(obs, vals):
    # fraction of shuffled values <= observed; lower is smoother
    return float(np.mean(np.asarray(vals) <= obs))

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

    rng = random.Random(SEED)
    summary_rows = []
    manifest_rows = []

    for (label, vid), rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda r: int(r["frame_num"]))
        arrays = []
        for r in rr:
            p = args.image_root / r["path"]
            if not p.exists():
                p = args.image_root / "images" / r["path"]
            arrays.append(load_gray(p))

        true_order = list(range(len(rr)))
        mae_matrix, ham_matrix = build_pairwise(arrays)
        obs_mae, obs_ham = mean_adjacent_distance(true_order, mae_matrix, ham_matrix)

        shuffle_mae, shuffle_ham = [], []
        for _ in range(N_SHUFFLES):
            order = true_order.copy()
            rng.shuffle(order)
            m,h = mean_adjacent_distance(order, mae_matrix, ham_matrix)
            shuffle_mae.append(m); shuffle_ham.append(h)

        z_mae, mu_mae, sd_mae = zscore(obs_mae, shuffle_mae)
        z_ham, mu_ham, sd_ham = zscore(obs_ham, shuffle_ham)
        pct_mae = percentile_lower(obs_mae, shuffle_mae)
        pct_ham = percentile_lower(obs_ham, shuffle_ham)

        frames = [int(r["frame_num"]) for r in rr]
        missing = [x for x in range(min(frames), max(frames)+1) if x not in set(frames)]
        summary_rows.append({
            "label": label,
            "video_id": vid,
            "file_count": len(rr),
            "unique_frame_count": len(set(frames)),
            "min_frame": min(frames),
            "max_frame": max(frames),
            "missing_frame_count": len(missing),
            "ordered_adjacent_gray_mae": obs_mae,
            "shuffle_mean_gray_mae": mu_mae,
            "shuffle_sd_gray_mae": sd_mae,
            "gray_mae_z_vs_shuffle": z_mae,
            "gray_mae_shuffle_percentile": pct_mae,
            "ordered_adjacent_dhash_hamming": obs_ham,
            "shuffle_mean_dhash_hamming": mu_ham,
            "shuffle_sd_dhash_hamming": sd_ham,
            "dhash_z_vs_shuffle": z_ham,
            "dhash_shuffle_percentile": pct_ham,
        })

        for pos, r in enumerate(rr):
            manifest_rows.append({
                "sequence_id": f"{label}__{vid}",
                "label": label,
                "video_id": vid,
                "sequence_position": pos,
                "frame_num": int(r["frame_num"]),
                "split": r["split"],
                "path": r["path"],
                "sha256": r["sha256"],
            })

    def write_csv(path, data):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    write_csv(args.out_dir / "ordered_sequence_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "temporal_order_vs_shuffle.csv", summary_rows)

    gray_strong = sum(r["gray_mae_shuffle_percentile"] <= 0.05 for r in summary_rows)
    hash_strong = sum(r["dhash_shuffle_percentile"] <= 0.05 for r in summary_rows)
    both_strong = sum(
        r["gray_mae_shuffle_percentile"] <= 0.05 and r["dhash_shuffle_percentile"] <= 0.05
        for r in summary_rows
    )
    aggregate = {
        "sequence_key": ["label","video_id"],
        "sequence_count": len(summary_rows),
        "total_files": len(rows),
        "shuffles_per_sequence": N_SHUFFLES,
        "random_seed": SEED,
        "metric_interpretation": "Lower adjacent distance means smoother visual continuity.",
        "sequences_ordered_smoother_than_95pct_shuffles_gray_mae": gray_strong,
        "sequences_ordered_smoother_than_95pct_shuffles_dhash": hash_strong,
        "sequences_ordered_smoother_than_95pct_shuffles_both_metrics": both_strong,
        "median_gray_percentile": float(np.median([r["gray_mae_shuffle_percentile"] for r in summary_rows])),
        "median_dhash_percentile": float(np.median([r["dhash_shuffle_percentile"] for r in summary_rows])),
        "mean_ordered_gray_mae": float(np.mean([r["ordered_adjacent_gray_mae"] for r in summary_rows])),
        "mean_shuffle_gray_mae": float(np.mean([r["shuffle_mean_gray_mae"] for r in summary_rows])),
        "mean_ordered_dhash_hamming": float(np.mean([r["ordered_adjacent_dhash_hamming"] for r in summary_rows])),
        "mean_shuffle_dhash_hamming": float(np.mean([r["shuffle_mean_dhash_hamming"] for r in summary_rows])),
        "scope_note": "This validates within-(label,VideoID) filename ordering against random shuffling. It does not establish real-world FPS or cross-label clip concatenation."
    }
    (args.out_dir / "temporal_order_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))

if __name__ == "__main__":
    main()
