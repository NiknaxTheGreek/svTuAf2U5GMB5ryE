#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash

BLOCK = 256

def hash_bits(h):
    # imagehash stores a boolean ndarray
    return np.asarray(h.hash, dtype=np.uint8).reshape(-1)

def popcount_matrix(a, b):
    # a: n x bits, b: m x bits
    # Hamming via boolean xor; block sizes keep memory modest
    return np.count_nonzero(a[:, None, :] != b[None, :, :], axis=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inventory.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    dh = np.zeros((n, 64), dtype=np.uint8)
    ph = np.zeros((n, 64), dtype=np.uint8)

    for i, r in enumerate(rows):
        p = args.image_root / r["path"]
        if not p.exists():
            p = args.image_root / "images" / r["path"]
        with Image.open(p) as im:
            im = im.convert("RGB")
            dh[i] = hash_bits(imagehash.dhash(im, hash_size=8))
            ph[i] = hash_bits(imagehash.phash(im, hash_size=8))

    # Track nearest neighbor outside own temporal clip and pair counts under conservative thresholds.
    nn_rows = []
    very_pairs = []
    moderate_pairs = []
    bucket_counts = Counter()

    seq_ids = np.array([f'{r["label"]}__{r["video_id"]}' for r in rows], dtype=object)
    splits = np.array([r["split"] for r in rows], dtype=object)
    labels = np.array([r["label"] for r in rows], dtype=object)

    best_score = np.full(n, 999, dtype=np.int16)
    best_j = np.full(n, -1, dtype=np.int32)
    best_d = np.full(n, 999, dtype=np.int16)
    best_p = np.full(n, 999, dtype=np.int16)

    def bucket(i,j):
        same_seq = seq_ids[i] == seq_ids[j]
        cross_split = splits[i] != splits[j]
        same_label = labels[i] == labels[j]
        if same_seq and cross_split: return "same_sequence_cross_split"
        if same_seq: return "same_sequence_same_split"
        if cross_split and same_label: return "different_sequence_cross_split_same_label"
        if cross_split and not same_label: return "different_sequence_cross_split_cross_label"
        if same_label: return "different_sequence_same_split_same_label"
        return "different_sequence_same_split_cross_label"

    for i0 in range(0, n, BLOCK):
        i1 = min(n, i0+BLOCK)
        dmat = popcount_matrix(dh[i0:i1], dh)
        pmat = popcount_matrix(ph[i0:i1], ph)

        for ii in range(i1-i0):
            i = i0 + ii
            # Ignore self and only evaluate j > i for pair summaries.
            score = dmat[ii] + pmat[ii]
            mask_outside = seq_ids != seq_ids[i]
            if np.any(mask_outside):
                cand = np.where(mask_outside, score, 999)
                j = int(np.argmin(cand))
                if cand[j] < best_score[i]:
                    best_score[i] = int(cand[j])
                    best_j[i] = j
                    best_d[i] = int(dmat[ii,j])
                    best_p[i] = int(pmat[ii,j])

            js = np.arange(n)
            pairmask = js > i
            # Very conservative: both hashes <=4
            vmask = pairmask & (dmat[ii] <= 4) & (pmat[ii] <= 4)
            for j in np.where(vmask)[0]:
                b = bucket(i,int(j))
                bucket_counts["very__"+b] += 1
                if len(very_pairs) < 500:
                    very_pairs.append({
                        "i": i, "j": int(j), "bucket": b,
                        "dhash": int(dmat[ii,j]), "phash": int(pmat[ii,j]),
                        "path_a": rows[i]["path"], "path_b": rows[int(j)]["path"],
                        "seq_a": seq_ids[i], "seq_b": seq_ids[int(j)],
                        "split_a": splits[i], "split_b": splits[int(j)],
                        "label_a": labels[i], "label_b": labels[int(j)],
                    })

            # Moderate: both hashes <=8
            mmask = pairmask & (dmat[ii] <= 8) & (pmat[ii] <= 8)
            for j in np.where(mmask)[0]:
                b = bucket(i,int(j))
                bucket_counts["moderate__"+b] += 1
                if len(moderate_pairs) < 1000:
                    moderate_pairs.append({
                        "i": i, "j": int(j), "bucket": b,
                        "dhash": int(dmat[ii,j]), "phash": int(pmat[ii,j]),
                        "path_a": rows[i]["path"], "path_b": rows[int(j)]["path"],
                        "seq_a": seq_ids[i], "seq_b": seq_ids[int(j)],
                        "split_a": splits[i], "split_b": splits[int(j)],
                        "label_a": labels[i], "label_b": labels[int(j)],
                    })

    for i in range(n):
        j = int(best_j[i])
        nn_rows.append({
            "path": rows[i]["path"],
            "sequence_id": seq_ids[i],
            "split": splits[i],
            "label": labels[i],
            "nearest_other_sequence_path": rows[j]["path"] if j >= 0 else "",
            "nearest_other_sequence_id": seq_ids[j] if j >= 0 else "",
            "nearest_other_sequence_split": splits[j] if j >= 0 else "",
            "nearest_other_sequence_label": labels[j] if j >= 0 else "",
            "dhash_distance": int(best_d[i]) if j >= 0 else "",
            "phash_distance": int(best_p[i]) if j >= 0 else "",
            "combined_distance": int(best_score[i]) if j >= 0 else "",
        })

    def write_csv(path, data):
        if not data:
            path.write_text("", encoding="utf-8"); return
        with path.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    write_csv(args.out_dir/"very_high_similarity_pairs.csv", very_pairs)
    write_csv(args.out_dir/"moderate_similarity_pairs.csv", moderate_pairs)
    write_csv(args.out_dir/"nearest_neighbor_outside_sequence.csv", nn_rows)

    # nearest-neighbor summary outside own sequence
    nn_score = np.array([r["combined_distance"] for r in nn_rows], dtype=float)
    nn_cross_split = sum(
        r["split"] != r["nearest_other_sequence_split"] for r in nn_rows if r["nearest_other_sequence_path"]
    )
    nn_cross_label = sum(
        r["label"] != r["nearest_other_sequence_label"] for r in nn_rows if r["nearest_other_sequence_path"]
    )

    summary = {
        "image_count": n,
        "total_unordered_pairs": n*(n-1)//2,
        "hashes": {"dhash_bits":64,"phash_bits":64},
        "very_high_similarity_rule": "dHash <= 4 AND pHash <= 4",
        "moderate_similarity_rule": "dHash <= 8 AND pHash <= 8",
        "pair_counts": dict(sorted(bucket_counts.items())),
        "nearest_neighbor_outside_sequence": {
            "median_combined_distance": float(np.median(nn_score)),
            "q1_combined_distance": float(np.quantile(nn_score,0.25)),
            "q3_combined_distance": float(np.quantile(nn_score,0.75)),
            "images_whose_nearest_other_sequence_neighbor_is_cross_split": int(nn_cross_split),
            "images_whose_nearest_other_sequence_neighbor_is_cross_label": int(nn_cross_label),
        },
        "scope_note": "Perceptual hashes are screening tools, not ground-truth duplicate labels. Similarity within one temporal clip is expected. Cross-sequence/cross-split matches are candidates for visual inspection; no image is deleted automatically."
    }
    (args.out_dir/"perceptual_similarity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
