from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from src.splits import (
    build_random_stratified_assignments,
    build_source_safe_assignments,
    build_source_safe_temporal_assignments,
    build_temporal_ordered_assignments,
    build_video_disjoint_assignments,
    load_environment_map,
    validate_dataset_split_compatibility,
    validate_environment_map,
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in ("flip", "notflip"):
        for video_num in range(1, 7):
            video_id = f"{label}/{video_num:04d}"
            for frame in range(1, 21):
                rows.append({
                    "sample_id": f"{video_id}_{frame:09d}",
                    "label": label,
                    "video_id": video_id,
                    "frame_number": str(frame),
                    "supplied_split": "training" if frame <= 16 else "testing",
                    "pixel_sha256": f"{len(rows):064x}"[-64:],
                })
    return rows


def _env_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in ("flip", "notflip"):
        for video_num in range(1, 7):
            env = "ENV-A" if video_num <= 2 else "ENV-B" if video_num <= 4 else "ENV-C"
            mapping[f"{label}/{video_num:04d}"] = env
    return mapping


def test_random_stratified_is_deterministic_and_three_way() -> None:
    rows = _rows()
    first = build_random_stratified_assignments(rows, seed=42)
    second = build_random_stratified_assignments(rows, seed=42)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert len(first) == len(rows)


def test_video_disjoint_has_no_group_overlap() -> None:
    rows = _rows()
    assignments = build_video_disjoint_assignments(rows, seed=42)
    by_video: dict[str, set[str]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], set()).add(assignments[row["sample_id"]])
    assert all(len(partitions) == 1 for partitions in by_video.values())
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_source_safe_is_environment_disjoint() -> None:
    rows = _rows()
    mapping = _env_map()
    assignments, env_partition = build_source_safe_assignments(rows, mapping)
    assert set(env_partition.values()) == {"train", "validation", "test"}
    by_env: dict[str, set[str]] = {}
    for row in rows:
        by_env.setdefault(mapping[row["video_id"]], set()).add(assignments[row["sample_id"]])
    assert all(len(partitions) == 1 for partitions in by_env.values())


def test_temporal_ordering_and_short_video_exclusion() -> None:
    rows = _rows()
    short = [{
        "sample_id": f"flip/9999_{frame:09d}",
        "label": "flip",
        "video_id": "flip/9999",
        "frame_number": str(frame),
        "supplied_split": "training",
        "pixel_sha256": f"{999000 + frame:064x}"[-64:],
    } for frame in range(1, 11)]
    assignments = build_temporal_ordered_assignments(rows + short)
    assert all(assignments[row["sample_id"]] == "excluded_short_video" for row in short)
    example = [row for row in rows if row["video_id"] == "flip/0001"]
    partitions = {name: [] for name in ("train", "validation", "test")}
    for row in example:
        partitions[assignments[row["sample_id"]]].append(int(row["frame_number"]))
    assert max(partitions["train"]) < min(partitions["validation"])
    assert max(partitions["validation"]) < min(partitions["test"])


def test_source_safe_temporal_holds_out_complete_environment() -> None:
    rows = _rows()
    mapping = _env_map()
    assignments, test_environment = build_source_safe_temporal_assignments(rows, mapping)
    test_rows = [row for row in rows if assignments[row["sample_id"]] == "test"]
    assert test_rows
    assert {mapping[row["video_id"]] for row in test_rows} == {test_environment}
    assert all(
        assignments[row["sample_id"]] == "test"
        for row in rows
        if mapping[row["video_id"]] == test_environment
    )


def test_environment_map_must_cover_manifest() -> None:
    rows = _rows()
    mapping = _env_map()
    mapping.pop(next(iter(mapping)))
    with pytest.raises(ValueError, match="Environment map mismatch"):
        validate_environment_map(rows, mapping)


def test_load_environment_map_rejects_duplicate_video(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    path.write_text(
        "video_id,environment_id,review_status\nflip/0001,ENV-A,reviewed\nflip/0001,ENV-B,reviewed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate video_id"):
        load_environment_map(path)


def test_dataset_split_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "compat.yaml"
    path.write_text(
        yaml.safe_dump({"compatibility": {"DATA-001": ["SPLIT-001", "SPLIT-002"]}}),
        encoding="utf-8",
    )
    validate_dataset_split_compatibility(path, dataset_id="DATA-001", split_id="SPLIT-001")
    with pytest.raises(ValueError, match="Incompatible"):
        validate_dataset_split_compatibility(path, dataset_id="DATA-001", split_id="SPLIT-999")


def test_frozen_environment_map_covers_real_manifest() -> None:
    manifest_path = Path("manifests/datasets/DATA-001.csv")
    map_path = Path("manifests/video_environment_map.csv")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = load_environment_map(map_path)
    validate_environment_map(rows, mapping)
    assert len(mapping) == 117
    assert set(mapping.values()) == {"ENV-01", "ENV-02", "ENV-03", "ENV-04"}
