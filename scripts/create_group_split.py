#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

PARTITIONS = ("train", "validation", "test")
TARGETS = {
    "flip": {
        "group_counts": {"train": 39, "validation": 13, "test": 13},
        "frame_counts": {"train": 872, "validation": 290, "test": 290},
    },
    "notflip": {
        "group_counts": {"train": 31, "validation": 10, "test": 11},
        "frame_counts": {"train": 923, "validation": 307, "test": 307},
    },
}


def stable_cost(group_id: str, partition: str) -> float:
    token = f"{group_id}|{partition}".encode("utf-8")
    value = int(hashlib.sha256(token).hexdigest()[:12], 16)
    return (value % 10_000_000) / 10_000_000


def solve_class(groups: pd.DataFrame, label: str) -> pd.DataFrame:
    subset = groups.loc[groups["label"] == label].sort_values("source_group_id").reset_index(drop=True)
    n = len(subset)
    m = len(PARTITIONS) * n

    objective = np.zeros(m, dtype=float)
    for i, group_id in enumerate(subset["source_group_id"]):
        for j, partition in enumerate(PARTITIONS):
            objective[j * n + i] = stable_cost(group_id, partition)

    rows, lower, upper = [], [], []

    # Each source group must belong to exactly one partition.
    for i in range(n):
        row = np.zeros(m, dtype=float)
        for j in range(len(PARTITIONS)):
            row[j * n + i] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    # Exact group counts and frame totals by class/partition.
    frame_counts = subset["frame_count"].to_numpy(dtype=float)
    for j, partition in enumerate(PARTITIONS):
        count_row = np.zeros(m, dtype=float)
        count_row[j * n : (j + 1) * n] = 1.0
        rows.append(count_row)
        target_group_count = TARGETS[label]["group_counts"][partition]
        lower.append(float(target_group_count))
        upper.append(float(target_group_count))

        frame_row = np.zeros(m, dtype=float)
        frame_row[j * n : (j + 1) * n] = frame_counts
        rows.append(frame_row)
        target_frame_count = TARGETS[label]["frame_counts"][partition]
        lower.append(float(target_frame_count))
        upper.append(float(target_frame_count))

    constraints = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
    result = milp(
        objective,
        integrality=np.ones(m),
        bounds=Bounds(np.zeros(m), np.ones(m)),
        constraints=constraints,
        options={"time_limit": 30},
    )
    if not result.success:
        raise RuntimeError(f"Could not construct split for {label}: {result.message}")

    chosen = np.rint(result.x).astype(int)
    partitions = []
    for i in range(n):
        hits = [j for j in range(len(PARTITIONS)) if chosen[j * n + i] == 1]
        if len(hits) != 1:
            raise AssertionError(f"Invalid assignment for {subset.loc[i, 'source_group_id']}: {hits}")
        partitions.append(PARTITIONS[hits[0]])

    subset = subset.copy()
    subset["partition"] = partitions
    return subset


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-groups", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    groups = pd.read_csv(args.source_groups, dtype={"video_id": str})
    frames = pd.read_csv(args.frames, dtype={"video_id_str": str})

    expected_group_columns = {"source_group_id", "label", "target", "video_id", "frame_count"}
    missing = expected_group_columns - set(groups.columns)
    if missing:
        raise ValueError(f"Source-group file missing columns: {sorted(missing)}")

    if len(groups) != 117 or set(groups["label"]) != {"flip", "notflip"}:
        raise AssertionError("Authoritative source-group population changed unexpectedly")

    assigned = pd.concat(
        [solve_class(groups, "flip"), solve_class(groups, "notflip")],
        ignore_index=True,
    ).sort_values("source_group_id").reset_index(drop=True)

    split_table = assigned[
        ["source_group_id", "label", "target", "video_id", "frame_count", "partition"]
    ].copy()

    split_sets = {
        p: set(split_table.loc[split_table["partition"] == p, "source_group_id"])
        for p in PARTITIONS
    }
    overlap_counts = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlap_counts.values()):
        raise AssertionError(f"Source-group leakage detected: {overlap_counts}")
    if set.union(*split_sets.values()) != set(groups["source_group_id"]):
        raise AssertionError("Split does not cover every source group exactly once")

    frame_split = frames.merge(
        split_table[["source_group_id", "partition"]],
        on="source_group_id",
        how="left",
        validate="many_to_one",
    )
    if frame_split["partition"].isna().any():
        raise AssertionError("At least one frame has no partition assignment")
    if len(frame_split) != 2989:
        raise AssertionError(f"Expected 2,989 frames, observed {len(frame_split)}")

    observed = (
        split_table.groupby(["partition", "label"])
        .agg(groups=("source_group_id", "size"), frames=("frame_count", "sum"))
        .reset_index()
    )
    for row in observed.itertuples(index=False):
        expected_groups = TARGETS[row.label]["group_counts"][row.partition]
        expected_frames = TARGETS[row.label]["frame_counts"][row.partition]
        if int(row.groups) != expected_groups or int(row.frames) != expected_frames:
            raise AssertionError(
                f"Split target mismatch for {row.partition}/{row.label}: "
                f"groups={row.groups}, frames={row.frames}"
            )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "source_group_split.csv"
    frame_path = out_dir / "frame_split.csv"
    split_table.to_csv(split_path, index=False)
    frame_split.to_csv(frame_path, index=False)

    summary = {
        "split_policy": "source-group-disjoint",
        "group_key": ["label", "video_id"],
        "selection_inputs": ["label", "source_group_id", "frame_count"],
        "model_or_pixel_information_used_for_split_selection": False,
        "ratio_target": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "group_counts": {
            str(k): int(v)
            for k, v in split_table.groupby("partition").size().items()
        },
        "frame_counts": {
            str(k): int(v)
            for k, v in frame_split.groupby("partition").size().items()
        },
        "by_partition_label": [
            {
                "partition": str(row.partition),
                "label": str(row.label),
                "groups": int(row.groups),
                "frames": int(row.frames),
            }
            for row in observed.itertuples(index=False)
        ],
        "flip_prevalence_by_partition": {
            p: float((frame_split.loc[frame_split["partition"] == p, "target"] == 1).mean())
            for p in PARTITIONS
        },
        "group_overlap_counts": overlap_counts,
        "source_group_split_sha256": sha256_file(split_path),
        "frame_split_sha256": sha256_file(frame_path),
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
