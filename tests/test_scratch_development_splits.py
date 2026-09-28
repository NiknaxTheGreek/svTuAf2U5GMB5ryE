from __future__ import annotations

from collections import defaultdict

from scripts.build_scratch_development_splits import (
    build_original_tuning_membership,
    build_source_disjoint_gate,
)


def _fixture_rows(tmp_path):
    path = tmp_path / "original.csv"
    lines = [
        "sample_id,label,video_id,frame_number,partition",
    ]
    for label in ("flip", "notflip"):
        for video in range(1, 13):
            for frame in range(1, 6):
                sample = f"{label}/{video:04d}_{frame:09d}"
                lines.append(f"{sample},{label},{label}/{video:04d},{frame},train")
        lines.append(f"{label}/9999_000000001,{label},{label}/9999,1,test")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_tuning_membership_is_stratified_and_excludes_test(tmp_path) -> None:
    path = _fixture_rows(tmp_path)
    rows = build_original_tuning_membership(path, seed=42)
    assert len(rows) == 120
    assert {row["partition"] for row in rows} == {"fit", "validation"}
    for label in ("flip", "notflip"):
        validation = [row for row in rows if row["label"] == label and row["partition"] == "validation"]
        assert len(validation) == 6


def test_source_gate_has_no_video_overlap(tmp_path) -> None:
    path = _fixture_rows(tmp_path)
    rows = build_source_disjoint_gate(path, seed=42)
    partitions = defaultdict(set)
    for row in rows:
        partitions[row["video_id"]].add(row["partition"])
    assert all(len(values) == 1 for values in partitions.values())
    assert {row["partition"] for row in rows} == {"fit", "validation"}
