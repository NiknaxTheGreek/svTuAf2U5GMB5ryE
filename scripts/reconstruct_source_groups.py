#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"split", "label", "path", "video_id", "frame_num"}
VALID_SPLITS = {"training", "testing"}
VALID_LABELS = {"flip", "notflip"}


def normalize_id(value) -> str:
    return f"{int(value):04d}"


def reconstruct(inventory_path: Path, out_dir: Path):
    frames = pd.read_csv(
        inventory_path,
        dtype={"video_id": "Int64", "frame_num": "Int64"},
    )
    missing = REQUIRED_COLUMNS - set(frames.columns)
    if missing:
        raise ValueError(f"Inventory missing required columns: {sorted(missing)}")
    if frames[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Inventory contains nulls in required reconstruction fields")
    if set(frames["split"].unique()) != VALID_SPLITS:
        raise ValueError(f"Unexpected split labels: {sorted(frames['split'].unique())}")
    if set(frames["label"].unique()) != VALID_LABELS:
        raise ValueError(f"Unexpected class labels: {sorted(frames['label'].unique())}")

    frames["video_id"] = frames["video_id"].astype(int)
    frames["frame_num"] = frames["frame_num"].astype(int)
    frames["video_id_str"] = frames["video_id"].map(normalize_id)
    frames["source_group_id"] = frames["label"] + "__" + frames["video_id_str"]
    frames["target"] = frames["label"].map({"notflip": 0, "flip": 1}).astype(int)

    duplicate_source_frames = frames.duplicated(
        subset=["label", "video_id", "frame_num"], keep=False
    )
    if duplicate_source_frames.any():
        sample = frames.loc[
            duplicate_source_frames,
            ["split", "label", "video_id", "frame_num", "path"],
        ].head(20)
        raise ValueError(
            "A source frame key appears more than once across supplied folders. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    group_rows = []
    for (label, video_id), g in frames.groupby(["label", "video_id"], sort=True):
        frame_numbers = sorted(g["frame_num"].tolist())
        min_frame, max_frame = frame_numbers[0], frame_numbers[-1]
        missing_frames = sorted(set(range(min_frame, max_frame + 1)) - set(frame_numbers))
        training_count = int((g["split"] == "training").sum())
        testing_count = int((g["split"] == "testing").sum())
        group_rows.append(
            {
                "source_group_id": f"{label}__{normalize_id(video_id)}",
                "label": label,
                "target": 1 if label == "flip" else 0,
                "video_id": normalize_id(video_id),
                "frame_count": len(g),
                "training_frame_count": training_count,
                "testing_frame_count": testing_count,
                "min_frame": min_frame,
                "max_frame": max_frame,
                "missing_frame_count": len(missing_frames),
                "missing_frames": ",".join(str(x) for x in missing_frames),
                "frame_sequence_contiguous": len(missing_frames) == 0,
                "present_in_supplied_training": training_count > 0,
                "present_in_supplied_testing": testing_count > 0,
                "crosses_supplied_split": training_count > 0 and testing_count > 0,
            }
        )

    groups = pd.DataFrame(group_rows).sort_values(
        ["target", "video_id"], ascending=[False, True]
    ).reset_index(drop=True)

    summary = {
        "input_inventory": str(inventory_path),
        "total_frames": int(len(frames)),
        "source_group_count": int(len(groups)),
        "source_groups_by_label": {
            str(k): int(v) for k, v in groups.groupby("label").size().items()
        },
        "frames_by_label": {
            str(k): int(v) for k, v in frames.groupby("label").size().items()
        },
        "groups_crossing_supplied_split": int(groups["crosses_supplied_split"].sum()),
        "groups_training_only": int(
            (groups["present_in_supplied_training"] & ~groups["present_in_supplied_testing"]).sum()
        ),
        "groups_testing_only": int(
            (~groups["present_in_supplied_training"] & groups["present_in_supplied_testing"]).sum()
        ),
        "groups_with_frame_gaps": int((~groups["frame_sequence_contiguous"]).sum()),
        "duplicate_source_frame_key_count": 0,
        "canonical_group_key": ["label", "video_id"],
        "target_mapping": {"notflip": 0, "flip": 1},
    }

    expected = {
        "total_frames": 2989,
        "source_group_count": 117,
        "flip_groups": 65,
        "notflip_groups": 52,
        "groups_crossing_supplied_split": 115,
        "groups_training_only": 2,
        "groups_testing_only": 0,
    }
    observed = {
        "total_frames": summary["total_frames"],
        "source_group_count": summary["source_group_count"],
        "flip_groups": summary["source_groups_by_label"].get("flip", 0),
        "notflip_groups": summary["source_groups_by_label"].get("notflip", 0),
        "groups_crossing_supplied_split": summary["groups_crossing_supplied_split"],
        "groups_training_only": summary["groups_training_only"],
        "groups_testing_only": summary["groups_testing_only"],
    }
    if observed != expected:
        raise AssertionError(f"Dataset lock mismatch. Expected {expected}, observed {observed}")

    out_dir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out_dir / "source_groups.csv", index=False)
    frames.sort_values(
        ["target", "video_id", "frame_num"], ascending=[False, True, True]
    ).to_csv(out_dir / "frames_with_source_group.csv", index=False)
    (out_dir / "source_group_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return frames, groups, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    _, groups, summary = reconstruct(args.inventory, args.out_dir)
    print(json.dumps(summary, indent=2))
    print("\nNon-contiguous source groups:")
    print(
        groups.loc[
            ~groups["frame_sequence_contiguous"],
            ["source_group_id", "frame_count", "min_frame", "max_frame", "missing_frames"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
