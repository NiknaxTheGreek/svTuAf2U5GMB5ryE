from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _stable_key(value: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def build_original_tuning_membership(
    original_split_path: str | Path,
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    with Path(original_split_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    training = [row for row in rows if row["partition"] == "train"]
    by_label: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in training:
        by_label[row["label"]].append(row)
    output: list[dict[str, str]] = []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda row: _stable_key(row["sample_id"], seed=seed))
        validation_n = max(1, int(__import__("math").ceil(0.10 * len(ordered))))
        validation_ids = {row["sample_id"] for row in ordered[-validation_n:]}
        for row in ordered:
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "video_id": row["video_id"],
                    "frame_number": row["frame_number"],
                    "partition": "validation" if row["sample_id"] in validation_ids else "fit",
                }
            )
    return sorted(output, key=lambda row: row["sample_id"])


def build_source_disjoint_gate(
    original_split_path: str | Path,
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    with Path(original_split_path).open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "train"]
    by_label_video: defaultdict[str, defaultdict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_label_video[row["label"]][row["video_id"]].append(row)

    validation_videos: set[str] = set()
    for label in sorted(by_label_video):
        groups = by_label_video[label]
        group_ids = sorted(groups)
        validation_group_n = max(1, int(__import__("math").ceil(0.10 * len(group_ids))))
        target_frames = 0.10 * sum(len(values) for values in groups.values())
        best: tuple[float, tuple[str, ...]] | None = None
        for offset in range(512):
            ordered = sorted(group_ids, key=lambda video_id: _stable_key(video_id, seed=seed + offset))
            selected = tuple(sorted(ordered[-validation_group_n:]))
            count = sum(len(groups[video_id]) for video_id in selected)
            candidate = (abs(count - target_frames), selected)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        validation_videos.update(best[1])

    output = []
    for row in rows:
        output.append(
            {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "video_id": row["video_id"],
                "frame_number": row["frame_number"],
                "partition": "validation" if row["video_id"] in validation_videos else "fit",
            }
        )
    return sorted(output, key=lambda row: row["sample_id"])


def write_membership(path: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "label", "video_id", "frame_number", "partition"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    original = Path("manifests/splits/SPLIT-001_original.csv")
    tuning = build_original_tuning_membership(original)
    gate = build_source_disjoint_gate(original)
    write_membership("manifests/splits/SPLIT-007_original_tuning.csv", tuning)
    write_membership("manifests/splits/SPLIT-008_original_source_gate.csv", gate)
    print(
        json.dumps(
            {
                "tuning": {
                    "fit": sum(row["partition"] == "fit" for row in tuning),
                    "validation": sum(row["partition"] == "validation" for row in tuning),
                },
                "source_gate": {
                    "fit": sum(row["partition"] == "fit" for row in gate),
                    "validation": sum(row["partition"] == "validation" for row in gate),
                    "validation_videos": len({row["video_id"] for row in gate if row["partition"] == "validation"}),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
