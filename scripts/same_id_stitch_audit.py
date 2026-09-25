#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RESIZE = (64, 64)
WINDOW = 3

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(RESIZE), dtype=np.float32) / 255.0

def dhash(arr: np.ndarray) -> np.ndarray:
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize((9, 8))
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)

def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))

def ham(a, b) -> float:
    return float(np.count_nonzero(dhash(a) != dhash(b)))

def percentile_lower(value: float, controls: list[float]) -> float:
    # fraction of control joins that are <= this distance; lower is more unusually smooth
    return float(np.mean(np.asarray(controls) <= value))

def endpoint_features(seq):
    k = min(WINDOW, len(seq))
    start = np.mean(np.stack(seq[:k]), axis=0)
    end = np.mean(np.stack(seq[-k:]), axis=0)
    return start, end

def adjacent_baseline(seq):
    if len(seq) < 2:
        return np.nan
    return float(np.median([mae(seq[i], seq[i+1]) for i in range(len(seq)-1)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["label"], r["video_id"])].append(r)

    seqs = {}
    meta = {}
    for key, rr in grouped.items():
        rr = sorted(rr, key=lambda r: int(r["frame_num"]))
        imgs = []
        for r in rr:
            p = args.image_root / r["path"]
            if not p.exists():
                p = args.image_root / "images" / r["path"]
            imgs.append(load_gray(p))
        seqs[key] = imgs
        meta[key] = rr

    paired_ids = sorted({
        vid for (label, vid) in grouped
        if ("flip", vid) in grouped and ("notflip", vid) in grouped
    })

    features = {}
    for key, seq in seqs.items():
        start, end = endpoint_features(seq)
        features[key] = {
            "start": start,
            "end": end,
            "within_adjacent_median_mae": adjacent_baseline(seq),
        }

    # Cross-ID controls for both directions.
    nf_ids = sorted(vid for (label, vid) in grouped if label == "notflip")
    f_ids = sorted(vid for (label, vid) in grouped if label == "flip")

    forward_controls = {}  # nf vid -> list joins to all other flip IDs
    reverse_controls = {}  # flip vid -> list joins to all other notflip IDs
    for vid in paired_ids:
        nf_end = features[("notflip", vid)]["end"]
        f_end = features[("flip", vid)]["end"]

        fc_mae, fc_ham = [], []
        for other in f_ids:
            if other == vid:
                continue
            fs = features[("flip", other)]["start"]
            fc_mae.append(mae(nf_end, fs))
            fc_ham.append(ham(nf_end, fs))
        forward_controls[vid] = (fc_mae, fc_ham)

        rc_mae, rc_ham = [], []
        for other in nf_ids:
            if other == vid:
                continue
            ns = features[("notflip", other)]["start"]
            rc_mae.append(mae(f_end, ns))
            rc_ham.append(ham(f_end, ns))
        reverse_controls[vid] = (rc_mae, rc_ham)

    out_rows = []
    for vid in paired_ids:
        nf = features[("notflip", vid)]
        fl = features[("flip", vid)]

        f_mae = mae(nf["end"], fl["start"])   # notflip -> flip
        f_ham = ham(nf["end"], fl["start"])
        r_mae = mae(fl["end"], nf["start"])   # flip -> notflip
        r_ham = ham(fl["end"], nf["start"])

        fc_mae, fc_ham = forward_controls[vid]
        rc_mae, rc_ham = reverse_controls[vid]

        f_pct_mae = percentile_lower(f_mae, fc_mae)
        f_pct_ham = percentile_lower(f_ham, fc_ham)
        r_pct_mae = percentile_lower(r_mae, rc_mae)
        r_pct_ham = percentile_lower(r_ham, rc_ham)

        # Lower composite percentile = more unusually smooth vs unrelated joins.
        f_composite = (f_pct_mae + f_pct_ham) / 2
        r_composite = (r_pct_mae + r_pct_ham) / 2

        within_ref = np.nanmedian([
            nf["within_adjacent_median_mae"],
            fl["within_adjacent_median_mae"],
        ])

        if f_composite < r_composite:
            preferred = "notflip->flip"
        elif r_composite < f_composite:
            preferred = "flip->notflip"
        else:
            preferred = "tie"

        out_rows.append({
            "video_id": vid,
            "notflip_frames": len(seqs[("notflip", vid)]),
            "flip_frames": len(seqs[("flip", vid)]),
            "within_sequence_median_adjacent_mae": float(within_ref),
            "notflip_to_flip_boundary_mae": f_mae,
            "notflip_to_flip_boundary_ratio_vs_within": float(f_mae / within_ref) if within_ref > 0 else np.nan,
            "notflip_to_flip_cross_id_percentile_mae": f_pct_mae,
            "notflip_to_flip_boundary_dhash": f_ham,
            "notflip_to_flip_cross_id_percentile_dhash": f_pct_ham,
            "notflip_to_flip_composite_percentile": f_composite,
            "flip_to_notflip_boundary_mae": r_mae,
            "flip_to_notflip_boundary_ratio_vs_within": float(r_mae / within_ref) if within_ref > 0 else np.nan,
            "flip_to_notflip_cross_id_percentile_mae": r_pct_mae,
            "flip_to_notflip_boundary_dhash": r_ham,
            "flip_to_notflip_cross_id_percentile_dhash": r_pct_ham,
            "flip_to_notflip_composite_percentile": r_composite,
            "preferred_direction_by_composite": preferred,
            "direction_margin": abs(f_composite - r_composite),
        })

    with (args.out_dir / "same_id_stitch_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    forward_better = sum(r["preferred_direction_by_composite"] == "notflip->flip" for r in out_rows)
    reverse_better = sum(r["preferred_direction_by_composite"] == "flip->notflip" for r in out_rows)
    ties = len(out_rows) - forward_better - reverse_better

    forward_strong = sum(
        r["notflip_to_flip_cross_id_percentile_mae"] <= 0.05 and
        r["notflip_to_flip_cross_id_percentile_dhash"] <= 0.05
        for r in out_rows
    )
    reverse_strong = sum(
        r["flip_to_notflip_cross_id_percentile_mae"] <= 0.05 and
        r["flip_to_notflip_cross_id_percentile_dhash"] <= 0.05
        for r in out_rows
    )
    either_strong = sum(
        (
            r["notflip_to_flip_cross_id_percentile_mae"] <= 0.05 and
            r["notflip_to_flip_cross_id_percentile_dhash"] <= 0.05
        ) or (
            r["flip_to_notflip_cross_id_percentile_mae"] <= 0.05 and
            r["flip_to_notflip_cross_id_percentile_dhash"] <= 0.05
        )
        for r in out_rows
    )

    summary = {
        "paired_same_numeric_video_ids": len(out_rows),
        "window_frames_per_endpoint": WINDOW,
        "interpretation": "Lower cross-ID percentile means the same-ID boundary is unusually smooth relative to unrelated joins.",
        "preferred_notflip_to_flip_count": forward_better,
        "preferred_flip_to_notflip_count": reverse_better,
        "ties": ties,
        "notflip_to_flip_strong_same_id_boundary_count": forward_strong,
        "flip_to_notflip_strong_same_id_boundary_count": reverse_strong,
        "either_direction_strong_same_id_boundary_count": either_strong,
        "median_notflip_to_flip_composite_percentile": float(np.median([r["notflip_to_flip_composite_percentile"] for r in out_rows])),
        "median_flip_to_notflip_composite_percentile": float(np.median([r["flip_to_notflip_composite_percentile"] for r in out_rows])),
        "median_notflip_to_flip_boundary_ratio_vs_within": float(np.median([r["notflip_to_flip_boundary_ratio_vs_within"] for r in out_rows])),
        "median_flip_to_notflip_boundary_ratio_vs_within": float(np.median([r["flip_to_notflip_boundary_ratio_vs_within"] for r in out_rows])),
        "scope_note": "This tests visual stitchability/session continuity only. Even a smooth join does not prove the two clips came from one uninterrupted original recording, and it does not recover original FPS."
    }
    (args.out_dir / "same_id_stitch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
