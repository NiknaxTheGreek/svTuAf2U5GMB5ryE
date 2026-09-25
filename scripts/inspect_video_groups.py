#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = read_rows(args.inventory)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # folder-defined groups: split + label + video_id
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["label"], r["video_id"])].append(int(r["frame_num"]))

    group_rows = []
    for (split, label, vid), frames in sorted(grouped.items()):
        uniq = sorted(set(frames))
        missing_inside = [n for n in range(min(uniq), max(uniq)+1) if n not in set(uniq)]
        group_rows.append({
            "split": split,
            "label": label,
            "video_id": vid,
            "frame_count": len(frames),
            "unique_frame_count": len(uniq),
            "min_frame": min(uniq),
            "max_frame": max(uniq),
            "span": max(uniq) - min(uniq) + 1,
            "missing_frame_count_inside_span": len(missing_inside),
        })
    write_csv(out / "folder_defined_video_groups.csv", group_rows)

    # candidate source groups: label + video_id, checking how supplied split partitions them
    source = defaultdict(lambda: {"training": [], "testing": []})
    for r in rows:
        source[(r["label"], r["video_id"])][r["split"]].append(int(r["frame_num"]))

    source_rows = []
    for (label, vid), d in sorted(source.items()):
        tr = sorted(set(d["training"]))
        te = sorted(set(d["testing"]))
        both = bool(tr and te)
        exact_overlap = sorted(set(tr) & set(te))
        combined = sorted(set(tr) | set(te))

        transitions = 0
        adjacent_cross_split_pairs = 0
        if both:
            membership = []
            trset, teset = set(tr), set(te)
            for f in combined:
                side = "both" if f in trset and f in teset else ("training" if f in trset else "testing")
                membership.append((f, side))
            last_side = None
            last_frame = None
            for f, side in membership:
                if side != "both" and last_side is not None and last_side != side:
                    transitions += 1
                    if last_frame is not None and f - last_frame == 1:
                        adjacent_cross_split_pairs += 1
                if side != "both":
                    last_side = side
                    last_frame = f

        source_rows.append({
            "label": label,
            "video_id": vid,
            "in_training": bool(tr),
            "in_testing": bool(te),
            "present_in_both_splits": both,
            "training_frame_count": len(tr),
            "testing_frame_count": len(te),
            "combined_frame_count": len(combined),
            "training_min_frame": min(tr) if tr else "",
            "training_max_frame": max(tr) if tr else "",
            "testing_min_frame": min(te) if te else "",
            "testing_max_frame": max(te) if te else "",
            "exact_frame_number_overlap_count": len(exact_overlap),
            "split_membership_transitions_by_frame_order": transitions,
            "adjacent_cross_split_frame_pairs": adjacent_cross_split_pairs,
        })
    write_csv(out / "candidate_source_groups.csv", source_rows)

    counts_by_folder = Counter((r["split"], r["label"]) for r in group_rows)
    frames_per_group = defaultdict(list)
    for r in group_rows:
        frames_per_group[(r["split"], r["label"])].append(int(r["frame_count"]))

    ids_by_split = defaultdict(set)
    ids_by_label_split = defaultdict(set)
    for r in rows:
        ids_by_split[r["split"]].add(r["video_id"])
        ids_by_label_split[(r["split"], r["label"])].add(r["video_id"])

    same_video_id_cross_label = {}
    for split in ("training", "testing"):
        flip_ids = ids_by_label_split[(split, "flip")]
        notflip_ids = ids_by_label_split[(split, "notflip")]
        same_video_id_cross_label[split] = sorted(flip_ids & notflip_ids)

    both = [r for r in source_rows if r["present_in_both_splits"]]
    training_only = [r for r in source_rows if r["in_training"] and not r["in_testing"]]
    testing_only = [r for r in source_rows if r["in_testing"] and not r["in_training"]]

    summary = {
        "input_image_rows": len(rows),
        "folder_defined_group_count": len(group_rows),
        "folder_defined_group_counts": {
            f"{split}|{label}": counts_by_folder[(split, label)]
            for split in ("training","testing")
            for label in ("flip","notflip")
        },
        "frames_per_folder_defined_group": {
            f"{split}|{label}": {
                "min": min(vals),
                "median": median(vals),
                "max": max(vals),
            }
            for (split,label), vals in sorted(frames_per_group.items())
        },
        "candidate_source_group_key": ["label", "video_id"],
        "candidate_source_group_count": len(source_rows),
        "candidate_source_groups_present_in_both_splits": len(both),
        "candidate_source_groups_training_only": len(training_only),
        "candidate_source_groups_testing_only": len(testing_only),
        "groups_with_exact_frame_number_overlap_across_splits": sum(
            int(r["exact_frame_number_overlap_count"]) > 0 for r in source_rows
        ),
        "groups_with_split_membership_transitions": sum(
            int(r["split_membership_transitions_by_frame_order"]) > 0 for r in source_rows
        ),
        "groups_with_adjacent_cross_split_frames": sum(
            int(r["adjacent_cross_split_frame_pairs"]) > 0 for r in source_rows
        ),
        "total_adjacent_cross_split_frame_pairs": sum(
            int(r["adjacent_cross_split_frame_pairs"]) for r in source_rows
        ),
        "same_video_id_used_in_both_labels_by_split": {
            split: {
                "count": len(ids),
                "examples": ids[:10],
            }
            for split, ids in same_video_id_cross_label.items()
        },
        "scope_note": "Fresh grouping inspection from raw image_inventory.csv only. No modeling or use of earlier group/split outputs."
    }
    (out / "video_group_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
