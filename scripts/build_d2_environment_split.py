#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_category(row: dict[str, str]) -> str:
    flip = int(row["flip_frames"])
    notflip = int(row["notflip_frames"])
    if flip and notflip:
        return "mixed"
    if flip:
        return "pure_flip"
    if notflip:
        return "pure_notflip"
    raise ValueError(f'Group {row["environment_group_id"]} has no frames')


def largest_remainder_targets(counts: dict[str, int], total_target: int) -> dict[str, int]:
    total = sum(counts.values())
    quotas = {k: counts[k] * total_target / total for k in sorted(counts)}
    out = {k: math.floor(v) for k, v in quotas.items()}
    remainder = total_target - sum(out.values())
    order = sorted(quotas, key=lambda k: (-(quotas[k] - out[k]), k))
    for k in order[:remainder]:
        out[k] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-summary", required=True, type=Path)
    ap.add_argument("--group-manifest", required=True, type=Path)
    ap.add_argument("--frame-manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = read_csv(args.group_summary)
    sequences = read_csv(args.group_manifest)
    frames = read_csv(args.frame_manifest)

    required_group_cols = {
        "environment_group_id", "group_type", "clip_count", "frame_count",
        "flip_frames", "notflip_frames", "training_frames", "testing_frames",
        "mixed_label_group", "labels", "sequences",
    }
    if not groups or not required_group_cols.issubset(groups[0]):
        raise SystemExit("Environment group summary schema mismatch")

    required_seq_cols = {
        "sequence_id", "label", "video_id", "environment_group_id",
        "group_type", "group_clip_count", "group_labels", "frame_count",
        "training_frames", "testing_frames",
    }
    if not sequences or not required_seq_cols.issubset(sequences[0]):
        raise SystemExit("Environment group manifest schema mismatch")

    required_frame_cols = {
        "sequence_id", "label", "video_id", "sequence_position",
        "frame_num", "split", "path", "sha256",
    }
    if not frames or not required_frame_cols.issubset(frames[0]):
        raise SystemExit("Ordered frame manifest schema mismatch")

    group_ids = [r["environment_group_id"] for r in groups]
    if len(group_ids) != len(set(group_ids)):
        raise SystemExit("Duplicate environment_group_id")
    seq_ids = [r["sequence_id"] for r in sequences]
    if len(seq_ids) != len(set(seq_ids)):
        raise SystemExit("Duplicate sequence_id")

    seq_to_group = {r["sequence_id"]: r["environment_group_id"] for r in sequences}
    if set(seq_to_group.values()) != set(group_ids):
        raise SystemExit("Sequence manifest and group summary disagree on group IDs")

    unknown_frame_sequences = sorted(set(r["sequence_id"] for r in frames) - set(seq_to_group))
    if unknown_frame_sequences:
        raise SystemExit(f"Frame manifest has unknown sequence IDs: {unknown_frame_sequences[:5]}")

    total_frames = sum(int(r["frame_count"]) for r in groups)
    total_flip = sum(int(r["flip_frames"]) for r in groups)
    total_notflip = sum(int(r["notflip_frames"]) for r in groups)
    if total_frames != len(frames) or total_frames != total_flip + total_notflip:
        raise SystemExit("Frame totals disagree across inputs")

    supplied_test_total = sum(r["split"] == "testing" for r in frames)
    supplied_test_flip = sum(
        r["split"] == "testing" and r["label"] == "flip" for r in frames
    )
    supplied_test_notflip = sum(
        r["split"] == "testing" and r["label"] == "notflip" for r in frames
    )
    if supplied_test_total != supplied_test_flip + supplied_test_notflip:
        raise SystemExit("Supplied test class totals do not reconcile")

    # D2 is intentionally matched to D1's supplied frame/class counts while
    # isolating entire conservative environment groups.
    test_group_target = round(len(groups) * supplied_test_total / total_frames)

    type_counts = Counter(r["group_type"] for r in groups)
    type_targets = largest_remainder_targets(dict(type_counts), test_group_target)
    if set(type_targets) != {"multi_clip", "singleton"}:
        raise SystemExit(f"Unexpected group types: {sorted(type_targets)}")

    category_counts = Counter(group_category(r) for r in groups)
    category_targets = largest_remainder_targets(dict(category_counts), test_group_target)
    expected_categories = {"pure_flip", "pure_notflip", "mixed"}
    if set(category_targets) != expected_categories:
        raise SystemExit(f"Unexpected group categories: {sorted(category_targets)}")

    target = (
        supplied_test_flip,
        supplied_test_notflip,
        test_group_target,
        type_targets["multi_clip"],
        category_targets["pure_flip"],
        category_targets["pure_notflip"],
        category_targets["mixed"],
    )

    # Exact bounded dynamic program. For every identical state, retain the
    # lexicographically smallest sorted group-ID tuple as the deterministic
    # tie-break.
    states: dict[tuple[int, ...], tuple[str, ...]] = {
        (0, 0, 0, 0, 0, 0, 0): ()
    }
    by_id = {r["environment_group_id"]: r for r in groups}

    for gid in sorted(by_id):
        r = by_id[gid]
        flip = int(r["flip_frames"])
        notflip = int(r["notflip_frames"])
        is_multi = int(r["group_type"] == "multi_clip")
        category = group_category(r)
        increments = (
            flip,
            notflip,
            1,
            is_multi,
            int(category == "pure_flip"),
            int(category == "pure_notflip"),
            int(category == "mixed"),
        )

        for state, selected in list(states.items()):
            new_state = tuple(state[i] + increments[i] for i in range(7))
            if any(new_state[i] > target[i] for i in range(7)):
                continue
            candidate = selected + (gid,)
            current = states.get(new_state)
            if current is None or candidate < current:
                states[new_state] = candidate

    selected_test = states.get(target)
    if selected_test is None:
        raise SystemExit(
            "No exact source-safe D2 split satisfies the frozen targets. "
            "Do not silently relax constraints; review the split policy."
        )

    test_ids = set(selected_test)
    train_ids = set(group_ids) - test_ids
    if test_ids & train_ids or test_ids | train_ids != set(group_ids):
        raise SystemExit("Environment group partition invariant failed")

    env_rows: list[dict[str, object]] = []
    for gid in sorted(group_ids):
        r = by_id[gid]
        env_rows.append({
            **r,
            "group_category": group_category(r),
            "d2_split": "test" if gid in test_ids else "train",
        })
    env_fields = list(groups[0].keys()) + ["group_category", "d2_split"]
    env_path = args.out_dir / "d2_environment_group_split.csv"
    write_csv(env_path, env_rows, env_fields)

    seq_rows: list[dict[str, object]] = []
    for r in sorted(sequences, key=lambda x: x["sequence_id"]):
        gid = r["environment_group_id"]
        seq_rows.append({
            **r,
            "group_category": group_category(by_id[gid]),
            "d2_split": "test" if gid in test_ids else "train",
        })
    seq_fields = list(sequences[0].keys()) + ["group_category", "d2_split"]
    seq_path = args.out_dir / "d2_sequence_split.csv"
    write_csv(seq_path, seq_rows, seq_fields)

    frame_rows: list[dict[str, object]] = []
    for r in frames:
        gid = seq_to_group[r["sequence_id"]]
        out = dict(r)
        out["supplied_split"] = out.pop("split")
        out["environment_group_id"] = gid
        out["d2_split"] = "test" if gid in test_ids else "train"
        frame_rows.append(out)
    frame_fields = [
        "sequence_id", "label", "video_id", "sequence_position", "frame_num",
        "supplied_split", "path", "sha256", "environment_group_id", "d2_split",
    ]
    frame_path = args.out_dir / "d2_frame_split_manifest.csv"
    write_csv(frame_path, frame_rows, frame_fields)

    def split_stats(split: str) -> dict[str, object]:
        selected_groups = [r for r in env_rows if r["d2_split"] == split]
        selected_sequences = [r for r in seq_rows if r["d2_split"] == split]
        selected_frames = [r for r in frame_rows if r["d2_split"] == split]
        return {
            "environment_groups": len(selected_groups),
            "multi_clip_groups": sum(r["group_type"] == "multi_clip" for r in selected_groups),
            "singleton_groups": sum(r["group_type"] == "singleton" for r in selected_groups),
            "pure_flip_groups": sum(r["group_category"] == "pure_flip" for r in selected_groups),
            "pure_notflip_groups": sum(r["group_category"] == "pure_notflip" for r in selected_groups),
            "mixed_groups": sum(r["group_category"] == "mixed" for r in selected_groups),
            "temporal_clips": len(selected_sequences),
            "frames": len(selected_frames),
            "flip_frames": sum(r["label"] == "flip" for r in selected_frames),
            "notflip_frames": sum(r["label"] == "notflip" for r in selected_frames),
        }

    train_stats = split_stats("train")
    test_stats = split_stats("test")

    # Hard acceptance gates.
    assert test_stats["frames"] == supplied_test_total
    assert test_stats["flip_frames"] == supplied_test_flip
    assert test_stats["notflip_frames"] == supplied_test_notflip
    assert test_stats["environment_groups"] == test_group_target
    assert test_stats["multi_clip_groups"] == type_targets["multi_clip"]
    assert test_stats["singleton_groups"] == type_targets["singleton"]
    assert test_stats["pure_flip_groups"] == category_targets["pure_flip"]
    assert test_stats["pure_notflip_groups"] == category_targets["pure_notflip"]
    assert test_stats["mixed_groups"] == category_targets["mixed"]
    assert train_stats["frames"] + test_stats["frames"] == total_frames

    train_seq = {r["sequence_id"] for r in seq_rows if r["d2_split"] == "train"}
    test_seq = {r["sequence_id"] for r in seq_rows if r["d2_split"] == "test"}
    train_paths = {r["path"] for r in frame_rows if r["d2_split"] == "train"}
    test_paths = {r["path"] for r in frame_rows if r["d2_split"] == "test"}

    overlap_checks = {
        "environment_group_overlap": len(train_ids & test_ids),
        "sequence_overlap": len(train_seq & test_seq),
        "frame_path_overlap": len(train_paths & test_paths),
    }
    if any(overlap_checks.values()):
        raise SystemExit(f"Split overlap invariant failed: {overlap_checks}")

    summary = {
        "dataset": "MonReader",
        "split_id": "D2",
        "split_name": "source_safe_all_images",
        "selection_policy": {
            "primary": "Match supplied D1 test frame and class counts exactly while keeping each frozen environment group indivisible.",
            "test_environment_group_target": test_group_target,
            "group_type_targets": type_targets,
            "group_category_targets": category_targets,
            "tie_break": "Lexicographically smallest sorted environment_group_id tuple among exact feasible solutions.",
            "relaxation_policy": "None. Fail if the exact frozen constraints become infeasible.",
        },
        "d1_reference_test": {
            "frames": supplied_test_total,
            "flip_frames": supplied_test_flip,
            "notflip_frames": supplied_test_notflip,
        },
        "d2": {
            "train": train_stats,
            "test": test_stats,
            "test_environment_group_ids": list(selected_test),
        },
        "overlap_checks": overlap_checks,
        "holdout_note": (
            "Environment-group construction used visual evidence before D2 assignment. "
            "After this assignment is frozen, the D2 test partition must not be used for "
            "model selection, threshold selection, debugging, or redesign."
        ),
        "d4_reuse_policy": (
            "D4 must reuse this exact environment-group train/test assignment and then "
            "apply the frozen deduplication mask within those assignments; D4 must not "
            "re-optimize the source split after deduplication."
        ),
    }
    summary_path = args.out_dir / "d2_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "inputs": {
            str(args.group_summary): sha256(args.group_summary),
            str(args.group_manifest): sha256(args.group_manifest),
            str(args.frame_manifest): sha256(args.frame_manifest),
        },
        "outputs": {
            env_path.name: sha256(env_path),
            seq_path.name: sha256(seq_path),
            frame_path.name: sha256(frame_path),
            summary_path.name: sha256(summary_path),
        },
        "status": "PASS",
    }
    receipt_path = args.out_dir / "d2_split_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
