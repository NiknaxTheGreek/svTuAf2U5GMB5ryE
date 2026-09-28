from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


DATASET_ID = "DATA-001"
PARTITIONS = ("train", "validation", "test")
SPLIT_DEFINITIONS = {
    "SPLIT-001": "Original",
    "SPLIT-002": "Random Stratified",
    "SPLIT-003": "Video-Disjoint",
    "SPLIT-004": "Source-Safe",
    "SPLIT-005": "Temporal-Ordered",
    "SPLIT-006": "Source-Safe + Temporal",
}


def _stable_key(value: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return sha256_file(output)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_environment_map(path: str | Path) -> dict[str, str]:
    rows = _read_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        video_id = row["video_id"]
        environment_id = row["environment_id"]
        if video_id in mapping:
            raise ValueError(f"Duplicate video_id in environment map: {video_id}")
        mapping[video_id] = environment_id
    if not mapping:
        raise ValueError("Environment map is empty")
    return mapping


def validate_environment_map(manifest_rows: Sequence[Mapping[str, str]], mapping: Mapping[str, str]) -> None:
    manifest_videos = {row["video_id"] for row in manifest_rows}
    mapping_videos = set(mapping)
    if manifest_videos != mapping_videos:
        missing = sorted(manifest_videos - mapping_videos)
        extra = sorted(mapping_videos - manifest_videos)
        raise ValueError(f"Environment map mismatch; missing={missing[:5]}, extra={extra[:5]}")


def build_original_assignments(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    result = {}
    for row in rows:
        supplied = row["supplied_split"]
        if supplied == "training":
            result[row["sample_id"]] = "train"
        elif supplied == "testing":
            result[row["sample_id"]] = "test"
        else:
            raise ValueError(f"Unexpected supplied split: {supplied}")
    return result


def _slice_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("At least three items are required for a three-way split")
    train_n = max(1, int(math.floor(ratios[0] * n)))
    validation_n = max(1, int(math.floor(ratios[1] * n)))
    test_n = n - train_n - validation_n
    if test_n < 1:
        test_n = 1
        if train_n >= validation_n and train_n > 1:
            train_n -= 1
        elif validation_n > 1:
            validation_n -= 1
        else:
            raise ValueError("Unable to allocate non-empty three-way split")
    return train_n, validation_n, test_n


def build_random_stratified_assignments(
    rows: Sequence[Mapping[str, str]], *, seed: int = 42
) -> dict[str, str]:
    by_label: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    assignments: dict[str, str] = {}
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda r: _stable_key(r["sample_id"], seed=seed))
        train_n, validation_n, _ = _slice_counts(len(ordered), (0.8, 0.1, 0.1))
        for index, row in enumerate(ordered):
            partition = "train" if index < train_n else "validation" if index < train_n + validation_n else "test"
            assignments[row["sample_id"]] = partition
    return assignments


def _best_group_split_for_label(
    groups: Mapping[str, int], *, seed: int, attempts: int = 512
) -> dict[str, str]:
    group_ids = sorted(groups)
    train_g, val_g, _ = _slice_counts(len(group_ids), (0.8, 0.1, 0.1))
    total = sum(groups.values())
    targets = {"train": 0.8 * total, "validation": 0.1 * total, "test": 0.1 * total}
    best_score: float | None = None
    best_assignment: dict[str, str] | None = None
    for offset in range(attempts):
        ordered = sorted(group_ids, key=lambda g: _stable_key(g, seed=seed + offset))
        candidate: dict[str, str] = {}
        for index, group_id in enumerate(ordered):
            candidate[group_id] = "train" if index < train_g else "validation" if index < train_g + val_g else "test"
        observed = Counter()
        for group_id, partition in candidate.items():
            observed[partition] += groups[group_id]
        score = sum(abs(observed[p] - targets[p]) / max(targets[p], 1.0) for p in PARTITIONS)
        if best_score is None or score < best_score or (
            math.isclose(score, best_score) and tuple(sorted(candidate.items())) < tuple(sorted(best_assignment.items()))
        ):
            best_score = score
            best_assignment = candidate
    assert best_assignment is not None
    return best_assignment


def build_video_disjoint_assignments(
    rows: Sequence[Mapping[str, str]], *, seed: int = 42
) -> dict[str, str]:
    videos_by_label: defaultdict[str, defaultdict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        videos_by_label[row["label"]][row["video_id"]].append(row)
    video_partition: dict[str, str] = {}
    for label in sorted(videos_by_label):
        group_sizes = {video_id: len(items) for video_id, items in videos_by_label[label].items()}
        video_partition.update(_best_group_split_for_label(group_sizes, seed=seed + len(video_partition)))
    return {row["sample_id"]: video_partition[row["video_id"]] for row in rows}


def choose_source_safe_environment_assignment(
    rows: Sequence[Mapping[str, str]], environment_map: Mapping[str, str]
) -> dict[str, str]:
    env_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts = Counter(row["label"] for row in rows)
    global_flip_fraction = global_counts["flip"] / sum(global_counts.values())
    for row in rows:
        env_counts[environment_map[row["video_id"]]][row["label"]] += 1
    if len(env_counts) < 3:
        raise ValueError("At least three environment/session groups are required")

    def balance_key(environment_id: str) -> tuple[float, str]:
        counts = env_counts[environment_id]
        fraction = counts["flip"] / sum(counts.values())
        return abs(fraction - global_flip_fraction), environment_id

    test_environment = min(env_counts, key=balance_key)
    remaining = [env for env in env_counts if env != test_environment]
    validation_environment = min(remaining, key=balance_key)
    assignment = {env: "train" for env in env_counts}
    assignment[test_environment] = "test"
    assignment[validation_environment] = "validation"
    return assignment


def build_source_safe_assignments(
    rows: Sequence[Mapping[str, str]], environment_map: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    env_partition = choose_source_safe_environment_assignment(rows, environment_map)
    assignments = {
        row["sample_id"]: env_partition[environment_map[row["video_id"]]] for row in rows
    }
    return assignments, env_partition


def _ordered_video_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["frame_number"]))
    frames = [int(row["frame_number"]) for row in ordered]
    if len(frames) != len(set(frames)):
        raise ValueError("Duplicate FrameNumber within video")
    return ordered


def build_temporal_ordered_assignments(
    rows: Sequence[Mapping[str, str]], *, minimum_frames: int = 20
) -> dict[str, str]:
    by_video: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    assignments: dict[str, str] = {}
    for video_id in sorted(by_video):
        ordered = _ordered_video_rows(by_video[video_id])
        if len(ordered) < minimum_frames:
            for row in ordered:
                assignments[row["sample_id"]] = "excluded_short_video"
            continue
        train_end = int(math.floor(0.8 * len(ordered)))
        validation_end = int(math.floor(0.9 * len(ordered)))
        if train_end < 1 or validation_end <= train_end or validation_end >= len(ordered):
            raise ValueError(f"Temporal split is not viable for {video_id}")
        for index, row in enumerate(ordered):
            partition = "train" if index < train_end else "validation" if index < validation_end else "test"
            assignments[row["sample_id"]] = partition
    return assignments


def build_source_safe_temporal_assignments(
    rows: Sequence[Mapping[str, str]],
    environment_map: Mapping[str, str],
    *,
    minimum_frames: int = 20,
) -> tuple[dict[str, str], str]:
    env_partition = choose_source_safe_environment_assignment(rows, environment_map)
    test_environment = next(env for env, partition in env_partition.items() if partition == "test")
    by_video: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    assignments: dict[str, str] = {}
    for video_id in sorted(by_video):
        ordered = _ordered_video_rows(by_video[video_id])
        environment_id = environment_map[video_id]
        if environment_id == test_environment:
            for row in ordered:
                assignments[row["sample_id"]] = "test"
            continue
        if len(ordered) < minimum_frames:
            for row in ordered:
                assignments[row["sample_id"]] = "excluded_short_video"
            continue
        train_end = int(math.floor(0.9 * len(ordered)))
        if train_end < 1 or train_end >= len(ordered):
            raise ValueError(f"Nested temporal split is not viable for {video_id}")
        for index, row in enumerate(ordered):
            assignments[row["sample_id"]] = "train" if index < train_end else "validation"
    return assignments, test_environment


def _split_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    assignments: Mapping[str, str],
    environment_map: Mapping[str, str],
    *,
    split_id: str,
    split_name: str,
) -> list[dict[str, object]]:
    output = []
    video_sizes = Counter(row["video_id"] for row in manifest_rows)
    for row in sorted(manifest_rows, key=lambda item: item["sample_id"]):
        sample_id = row["sample_id"]
        if sample_id not in assignments:
            raise ValueError(f"Missing partition for {sample_id}")
        output.append(
            {
                "sample_id": sample_id,
                "dataset_id": DATASET_ID,
                "split_id": split_id,
                "split_name": split_name,
                "partition": assignments[sample_id],
                "label": row["label"],
                "video_id": row["video_id"],
                "environment_id": environment_map[row["video_id"]],
                "frame_number": int(row["frame_number"]),
                "supplied_split": row["supplied_split"],
                "pixel_sha256": row["pixel_sha256"],
                "temporal_eligible": str(video_sizes[row["video_id"]] >= 20).lower(),
            }
        )
    return output


def summarize_split(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    partition_counts = Counter(str(row["partition"]) for row in rows)
    partition_label_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    partition_video_counts: defaultdict[str, set[str]] = defaultdict(set)
    partition_environment_counts: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        partition = str(row["partition"])
        partition_label_counts[partition][str(row["label"])] += 1
        partition_video_counts[partition].add(str(row["video_id"]))
        partition_environment_counts[partition].add(str(row["environment_id"]))
    return {
        "row_count": len(rows),
        "partition_counts": dict(sorted(partition_counts.items())),
        "partition_label_counts": {
            partition: dict(sorted(counts.items()))
            for partition, counts in sorted(partition_label_counts.items())
        },
        "partition_video_counts": {
            partition: len(values) for partition, values in sorted(partition_video_counts.items())
        },
        "partition_environment_counts": {
            partition: len(values) for partition, values in sorted(partition_environment_counts.items())
        },
    }


def validate_split(rows: Sequence[Mapping[str, object]], *, split_name: str) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample assignment in {split_name}")
    active = [row for row in rows if str(row["partition"]) in PARTITIONS]
    if split_name in {"Video-Disjoint", "Source-Safe"}:
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for row in active:
            seen[str(row["video_id"])].add(str(row["partition"]))
        bad = [video_id for video_id, partitions in seen.items() if len(partitions) > 1]
        if bad:
            raise ValueError(f"Video leakage in {split_name}: {bad[:5]}")
    if split_name == "Source-Safe":
        seen_env: defaultdict[str, set[str]] = defaultdict(set)
        for row in active:
            seen_env[str(row["environment_id"])].add(str(row["partition"]))
        bad = [env for env, partitions in seen_env.items() if len(partitions) > 1]
        if bad:
            raise ValueError(f"Environment leakage in Source-Safe: {bad}")
    if split_name == "Temporal-Ordered":
        by_video: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for row in active:
            by_video[str(row["video_id"])][str(row["partition"])].append(int(row["frame_number"]))
        for video_id, partitions in by_video.items():
            if set(partitions) != set(PARTITIONS):
                raise ValueError(f"Temporal partitions incomplete for {video_id}")
            if not (max(partitions["train"]) < min(partitions["validation"]) <= max(partitions["validation"]) < min(partitions["test"])):
                raise ValueError(f"Temporal chronology violated for {video_id}")
    if split_name == "Source-Safe + Temporal":
        test_envs = {str(row["environment_id"]) for row in active if row["partition"] == "test"}
        non_test_envs = {
            str(row["environment_id"])
            for row in active
            if row["partition"] in {"train", "validation"}
        }
        if test_envs & non_test_envs:
            raise ValueError("Source-Safe + Temporal test environment leaks into fitting pool")
        by_video: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for row in active:
            if row["partition"] in {"train", "validation"}:
                by_video[str(row["video_id"])][str(row["partition"])].append(int(row["frame_number"]))
        for video_id, partitions in by_video.items():
            if set(partitions) != {"train", "validation"}:
                raise ValueError(f"Nested temporal partitions incomplete for {video_id}")
            if max(partitions["train"]) >= min(partitions["validation"]):
                raise ValueError(f"Nested temporal chronology violated for {video_id}")


def _registration(
    *,
    split_id: str,
    split_name: str,
    manifest_relpath: str,
    manifest_sha256: str,
    summary: Mapping[str, object],
    rule: Mapping[str, object],
) -> dict[str, object]:
    return {
        "split_id": split_id,
        "name": split_name,
        "dataset_id": DATASET_ID,
        "status": "canonical",
        "manifest": {
            "path": manifest_relpath,
            "sha256": manifest_sha256,
            "row_count": summary["row_count"],
        },
        "summary": dict(summary),
        "rule": dict(rule),
    }


def validate_dataset_split_compatibility(
    registry_path: str | Path, *, dataset_id: str, split_id: str
) -> None:
    payload = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    allowed = payload.get("compatibility", {})
    split_ids = set(allowed.get(dataset_id, []))
    if split_id not in split_ids:
        raise ValueError(f"Incompatible dataset/split pair: {dataset_id} + {split_id}")


def build_split_artifacts(
    *,
    manifest_path: str | Path,
    environment_map_path: str | Path,
    output_root: str | Path,
    seed: int = 42,
) -> dict[str, object]:
    manifest_rows = _read_csv(manifest_path)
    environment_map = load_environment_map(environment_map_path)
    validate_environment_map(manifest_rows, environment_map)

    definitions: list[tuple[str, str, dict[str, str], dict[str, object]]] = []
    definitions.append((
        "SPLIT-001",
        "Original",
        build_original_assignments(manifest_rows),
        {"type": "supplied", "training": "train", "testing": "test", "validation": None},
    ))
    definitions.append((
        "SPLIT-002",
        "Random Stratified",
        build_random_stratified_assignments(manifest_rows, seed=seed),
        {"type": "frame_stratified", "ratios": [0.8, 0.1, 0.1], "seed": seed},
    ))
    definitions.append((
        "SPLIT-003",
        "Video-Disjoint",
        build_video_disjoint_assignments(manifest_rows, seed=seed),
        {"type": "video_disjoint", "ratios": [0.8, 0.1, 0.1], "seed": seed, "group": "video_id"},
    ))
    source_assignments, env_partition = build_source_safe_assignments(manifest_rows, environment_map)
    definitions.append((
        "SPLIT-004",
        "Source-Safe",
        source_assignments,
        {
            "type": "environment_session_disjoint",
            "group": "environment_id",
            "environment_partition": env_partition,
            "selection": "test environment closest to global class proportion; validation chosen likewise from remaining groups",
        },
    ))
    definitions.append((
        "SPLIT-005",
        "Temporal-Ordered",
        build_temporal_ordered_assignments(manifest_rows),
        {"type": "within_video_chronological", "ratios": [0.8, 0.1, 0.1], "minimum_frames": 20},
    ))
    st_assignments, st_test_environment = build_source_safe_temporal_assignments(
        manifest_rows, environment_map
    )
    definitions.append((
        "SPLIT-006",
        "Source-Safe + Temporal",
        st_assignments,
        {
            "type": "nested_environment_holdout_plus_temporal",
            "test_environment": st_test_environment,
            "inner_train_validation_ratios": [0.9, 0.1],
            "minimum_frames_for_inner_temporal": 20,
            "outer_test_group_is_complete": True,
        },
    ))

    output_root = Path(output_root)
    split_dir = output_root / "manifests" / "splits"
    fieldnames = [
        "sample_id", "dataset_id", "split_id", "split_name", "partition", "label",
        "video_id", "environment_id", "frame_number", "supplied_split", "pixel_sha256",
        "temporal_eligible",
    ]
    result: dict[str, object] = {"splits": {}}
    for split_id, split_name, assignments, rule in definitions:
        split_rows = _split_rows(
            manifest_rows, assignments, environment_map, split_id=split_id, split_name=split_name
        )
        validate_split(split_rows, split_name=split_name)
        stem = f"{split_id}_{split_name.lower().replace(' + ', '_').replace('-', '_').replace(' ', '_')}"
        manifest_output = split_dir / f"{stem}.csv"
        digest = _write_csv(manifest_output, split_rows, fieldnames)
        summary = summarize_split(split_rows)
        registration = _registration(
            split_id=split_id,
            split_name=split_name,
            manifest_relpath=(Path("manifests") / "splits" / manifest_output.name).as_posix(),
            manifest_sha256=digest,
            summary=summary,
            rule=rule,
        )
        (split_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump(registration, sort_keys=False), encoding="utf-8"
        )
        result["splits"][split_id] = registration

    compatibility = {"compatibility": {DATASET_ID: list(SPLIT_DEFINITIONS)}}
    compatibility_path = output_root / "manifests" / "dataset_split_compatibility.yaml"
    compatibility_path.parent.mkdir(parents=True, exist_ok=True)
    compatibility_path.write_text(yaml.safe_dump(compatibility, sort_keys=False), encoding="utf-8")
    result["compatibility"] = compatibility
    (split_dir / "split_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
