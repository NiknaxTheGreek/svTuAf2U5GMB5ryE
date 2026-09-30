from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_MASTER_ROWS = 2989
EXPECTED_MASTER_SHA256 = "3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b"
HELDOUT_ENVIRONMENT = "ENV-03"
VALID_ENVIRONMENTS = {"ENV-01", "ENV-02", "ENV-03", "ENV-04"}
ROLE_ORDER = {"train": 0, "test": 1, "context": 2, "excluded": 3}
FIELDS = [
    "regime",
    "sample_id",
    "role",
    "role_detail",
    "label",
    "video_id",
    "frame_number",
    "environment_id",
    "supplied_split",
    "video_observed_length",
    "temporal_rank_0based",
    "temporal_cut_floor80",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in FIELDS})


def load_inputs(manifest_path: Path, environment_map_path: Path):
    if sha256_file(manifest_path) != EXPECTED_MASTER_SHA256:
        raise RuntimeError(
            f"Master manifest SHA-256 mismatch: {sha256_file(manifest_path)} != {EXPECTED_MASTER_SHA256}"
        )

    manifest = read_csv(manifest_path)
    if len(manifest) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_MASTER_ROWS} master rows, found {len(manifest)}")

    required = {
        "sample_id", "supplied_split", "label", "video_id", "frame_number", "archive_member"
    }
    if not manifest or required - set(manifest[0]):
        raise RuntimeError(f"Master manifest missing required columns: {sorted(required - set(manifest[0]))}")

    sample_ids = [r["sample_id"] for r in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Master sample_id values are not unique")

    env_rows = read_csv(environment_map_path)
    if not env_rows:
        raise RuntimeError("Environment map is empty")
    env_ids = [r["video_id"] for r in env_rows]
    if len(env_ids) != len(set(env_ids)):
        raise RuntimeError("Environment map has duplicate video_id values")

    env_by_video = {r["video_id"]: r["environment_id"] for r in env_rows}
    master_videos = {r["video_id"] for r in manifest}
    if set(env_by_video) != master_videos:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_environment_rows": sorted(master_videos - set(env_by_video)),
                    "extra_environment_rows": sorted(set(env_by_video) - master_videos),
                },
                indent=2,
            )
        )
    invalid_envs = set(env_by_video.values()) - VALID_ENVIRONMENTS
    if invalid_envs:
        raise RuntimeError(f"Invalid environment IDs: {sorted(invalid_envs)}")

    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in manifest:
        x = dict(r)
        x["frame_number_int"] = int(r["frame_number"])
        x["environment_id"] = env_by_video[r["video_id"]]
        by_video[r["video_id"]].append(x)

    temporal_meta: dict[str, dict[str, int]] = {}
    for video_id, group in by_video.items():
        group.sort(key=lambda r: (r["frame_number_int"], r["sample_id"]))
        frames = [r["frame_number_int"] for r in group]
        if len(frames) < 2:
            raise RuntimeError(f"Video {video_id} cannot support a chronological train/test split")
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise RuntimeError(f"Non-increasing or duplicate FrameNumber in {video_id}")
        cut = math.floor(0.80 * len(group))
        if not 1 <= cut < len(group):
            raise RuntimeError(f"Invalid 80/20 cut for {video_id}: n={len(group)}, cut={cut}")
        temporal_meta[video_id] = {"length": len(group), "cut": cut}
        for rank, r in enumerate(group):
            r["temporal_rank_0based"] = rank
            r["video_observed_length"] = len(group)
            r["temporal_cut_floor80"] = cut

    flat = [r for video_id in sorted(by_video) for r in by_video[video_id]]
    return flat, by_video, temporal_meta


def base_record(regime: str, r: dict[str, object], role: str, role_detail: str) -> dict[str, object]:
    return {
        "regime": regime,
        "sample_id": r["sample_id"],
        "role": role,
        "role_detail": role_detail,
        "label": r["label"],
        "video_id": r["video_id"],
        "frame_number": r["frame_number_int"],
        "environment_id": r["environment_id"],
        "supplied_split": r["supplied_split"],
        "video_observed_length": r["video_observed_length"],
        "temporal_rank_0based": r["temporal_rank_0based"],
        "temporal_cut_floor80": r["temporal_cut_floor80"],
    }


def build_regimes(flat: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    regimes: dict[str, list[dict[str, object]]] = {k: [] for k in ("O", "S", "T", "ST")}

    for r in flat:
        rank = int(r["temporal_rank_0based"])
        cut = int(r["temporal_cut_floor80"])
        env = str(r["environment_id"])

        if r["supplied_split"] == "training":
            regimes["O"].append(base_record("O", r, "train", "supplied_training"))
        elif r["supplied_split"] == "testing":
            regimes["O"].append(base_record("O", r, "test", "supplied_testing"))
        else:
            raise RuntimeError(f"Unexpected supplied split: {r['supplied_split']}")

        if env == HELDOUT_ENVIRONMENT:
            regimes["S"].append(base_record("S", r, "test", "heldout_environment_test"))
        else:
            regimes["S"].append(base_record("S", r, "train", "known_environment_train"))

        if rank < cut:
            regimes["T"].append(base_record("T", r, "train", "earlier_80pct_train"))
        else:
            regimes["T"].append(base_record("T", r, "test", "later_20pct_test"))

        if env == HELDOUT_ENVIRONMENT:
            if rank < cut:
                regimes["ST"].append(base_record("ST", r, "context", "heldout_environment_earlier_context"))
            else:
                regimes["ST"].append(base_record("ST", r, "test", "heldout_environment_later_test"))
        else:
            if rank < cut:
                regimes["ST"].append(base_record("ST", r, "train", "known_environment_earlier_train"))
            else:
                regimes["ST"].append(base_record("ST", r, "context", "known_environment_later_context"))

    for regime in regimes:
        regimes[regime].sort(key=lambda r: (r["video_id"], int(r["frame_number"]), r["sample_id"]))
    return regimes


def assert_complete(name: str, rows: list[dict[str, object]], master_ids: set[str]) -> None:
    ids = [str(r["sample_id"]) for r in rows]
    if len(rows) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(f"{name}: expected {EXPECTED_MASTER_ROWS} rows, found {len(rows)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{name}: duplicate sample IDs")
    if set(ids) != master_ids:
        raise RuntimeError(
            f"{name}: sample accounting mismatch: missing={len(master_ids-set(ids))}, extra={len(set(ids)-master_ids)}"
        )
    if any(r["role"] == "excluded" for r in rows):
        raise RuntimeError(f"{name}: exclusions are not permitted in the primary split design")


def assert_temporal_boundary(rows: list[dict[str, object]], train_role: str, test_role: str) -> None:
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_video[str(r["video_id"])].append(r)
    for video_id, group in by_video.items():
        a = [int(r["frame_number"]) for r in group if r["role"] == train_role]
        b = [int(r["frame_number"]) for r in group if r["role"] == test_role]
        if a and b and max(a) >= min(b):
            raise RuntimeError(f"Temporal ordering violated in {video_id}: max earlier={max(a)}, min later={min(b)}")


def audit_regimes(regimes: dict[str, list[dict[str, object]]], master_ids: set[str]) -> dict:
    for name, rows in regimes.items():
        assert_complete(name, rows, master_ids)

    # O exactly reproduces supplied membership.
    for r in regimes["O"]:
        expected = "train" if r["supplied_split"] == "training" else "test"
        if r["role"] != expected:
            raise RuntimeError(f"O supplied-split mismatch for {r['sample_id']}")

    # S is environment-disjoint.
    for r in regimes["S"]:
        if (r["role"] == "test") != (r["environment_id"] == HELDOUT_ENVIRONMENT):
            raise RuntimeError(f"S environment-role mismatch for {r['sample_id']}")
    s_train_videos = {r["video_id"] for r in regimes["S"] if r["role"] == "train"}
    s_test_videos = {r["video_id"] for r in regimes["S"] if r["role"] == "test"}
    if s_train_videos & s_test_videos:
        raise RuntimeError("S train/test video overlap is non-zero")

    # T uses every video on both chronological sides.
    t_by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in regimes["T"]:
        t_by_video[str(r["video_id"])].append(r)
    for video_id, group in t_by_video.items():
        roles = {r["role"] for r in group}
        if roles != {"train", "test"}:
            raise RuntimeError(f"T video {video_id} lacks both train and test roles")
        train_ranks = [int(r["temporal_rank_0based"]) for r in group if r["role"] == "train"]
        test_ranks = [int(r["temporal_rank_0based"]) for r in group if r["role"] == "test"]
        if max(train_ranks) >= min(test_ranks):
            raise RuntimeError(f"T chronological rank boundary violated for {video_id}")
    assert_temporal_boundary(regimes["T"], "train", "test")

    # ST: no held-out environment in training; primary test is later ENV-03 only.
    st = regimes["ST"]
    if any(r["environment_id"] == HELDOUT_ENVIRONMENT for r in st if r["role"] == "train"):
        raise RuntimeError("ST training contains held-out ENV-03")
    if any(r["environment_id"] != HELDOUT_ENVIRONMENT for r in st if r["role"] == "test"):
        raise RuntimeError("ST primary test contains a known environment")
    st_train_videos = {r["video_id"] for r in st if r["role"] == "train"}
    st_test_videos = {r["video_id"] for r in st if r["role"] == "test"}
    if st_train_videos & st_test_videos:
        raise RuntimeError("ST train/test video overlap is non-zero")
    for r in st:
        rank = int(r["temporal_rank_0based"])
        cut = int(r["temporal_cut_floor80"])
        if r["role_detail"] == "heldout_environment_earlier_context" and not (
            r["environment_id"] == HELDOUT_ENVIRONMENT and rank < cut
        ):
            raise RuntimeError(f"ST heldout-earlier context mismatch: {r['sample_id']}")
        if r["role_detail"] == "heldout_environment_later_test" and not (
            r["environment_id"] == HELDOUT_ENVIRONMENT and rank >= cut
        ):
            raise RuntimeError(f"ST heldout-later test mismatch: {r['sample_id']}")
        if r["role_detail"] == "known_environment_earlier_train" and not (
            r["environment_id"] != HELDOUT_ENVIRONMENT and rank < cut
        ):
            raise RuntimeError(f"ST known-earlier train mismatch: {r['sample_id']}")
        if r["role_detail"] == "known_environment_later_context" and not (
            r["environment_id"] != HELDOUT_ENVIRONMENT and rank >= cut
        ):
            raise RuntimeError(f"ST known-later context mismatch: {r['sample_id']}")

    expected_counts = {
        "O": {"train": 2392, "test": 597},
        "S": {"train": 2219, "test": 770},
        "T": {"train": 2365, "test": 624},
        "ST": {"train": 1756, "test": 161, "context": 1072},
    }
    summaries = {}
    for name, rows in regimes.items():
        role_counts = Counter(str(r["role"]) for r in rows)
        if dict(role_counts) != expected_counts[name]:
            raise RuntimeError(f"{name}: unexpected role counts {dict(role_counts)} != {expected_counts[name]}")
        role_detail_counts = Counter(str(r["role_detail"]) for r in rows)
        class_by_role = Counter((str(r["role"]), str(r["label"])) for r in rows)
        env_by_role = Counter((str(r["role"]), str(r["environment_id"])) for r in rows)
        train_videos = {r["video_id"] for r in rows if r["role"] == "train"}
        test_videos = {r["video_id"] for r in rows if r["role"] == "test"}
        summaries[name] = {
            "total_accounted": len(rows),
            "role_counts": dict(sorted(role_counts.items())),
            "role_detail_counts": dict(sorted(role_detail_counts.items())),
            "class_counts_by_role": {
                f"{role}/{label}": n for (role, label), n in sorted(class_by_role.items())
            },
            "environment_counts_by_role": {
                f"{role}/{env}": n for (role, env), n in sorted(env_by_role.items())
            },
            "train_video_count": len(train_videos),
            "test_video_count": len(test_videos),
            "train_test_video_overlap_count": len(train_videos & test_videos),
            "excluded": 0,
        }

    # Explicit ST context decomposition.
    st_details = Counter(str(r["role_detail"]) for r in regimes["ST"])
    if st_details["heldout_environment_earlier_context"] != 609:
        raise RuntimeError("ST earlier held-out context sanity check failed")
    if st_details["known_environment_later_context"] != 463:
        raise RuntimeError("ST later known-source context sanity check failed")

    return summaries


def report_markdown(summary: dict) -> str:
    s = summary["regimes"]
    return f"""# MonReader V2 Split Audit

## Frozen split rules

All splits are derived from the verified 2,989-row `DATA-001` manifest.

For temporal regimes, each canonical video is sorted by `FrameNumber`. With `n` observed images:

`cut = floor(0.80 × n)`

Rows with temporal rank below `cut` are the earlier portion; rows at or above `cut` are the later portion. The rule operates on observed images and does not invent missing frame numbers.

Source-safe holdout: **{HELDOUT_ENVIRONMENT}**.

## Accounting

| Regime | Training | Primary test | Context-only | Excluded | Total |
|---|---:|---:|---:|---:|---:|
| O | {s['O']['role_counts'].get('train',0)} | {s['O']['role_counts'].get('test',0)} | 0 | 0 | {s['O']['total_accounted']} |
| S | {s['S']['role_counts'].get('train',0)} | {s['S']['role_counts'].get('test',0)} | 0 | 0 | {s['S']['total_accounted']} |
| T | {s['T']['role_counts'].get('train',0)} | {s['T']['role_counts'].get('test',0)} | 0 | 0 | {s['T']['total_accounted']} |
| ST | {s['ST']['role_counts'].get('train',0)} | {s['ST']['role_counts'].get('test',0)} | {s['ST']['role_counts'].get('context',0)} | 0 | {s['ST']['total_accounted']} |

ST context-only decomposition:
- earlier held-out {HELDOUT_ENVIRONMENT}: {s['ST']['role_detail_counts'].get('heldout_environment_earlier_context',0)}
- later known-source environments: {s['ST']['role_detail_counts'].get('known_environment_later_context',0)}

## Scientific meaning

**O — Original:** supplied training vs supplied testing exactly as distributed. Its test is source-joint: all 597 test images are from videos represented in supplied training.

**S — Source-Safe:** all {HELDOUT_ENVIRONMENT} images are test; all ENV-01/02/04 images are training. Train/test canonical-video overlap is zero.

**T — Temporal:** every video contributes its earlier observed 80% to training and its later observed 20% to test. This measures future-frame generalization within known sources.

**ST — Source-Safe + Temporal:** ENV-03 is completely absent from training. For ENV-03, earlier frames are context-only and later frames form the primary test. For known environments, earlier frames train and later frames are context-only. Train/test canonical-video overlap is zero.

## Leakage/accounting gates

- Every regime accounts for exactly 2,989 unique sample IDs.
- Excluded samples: 0 in every regime.
- S and ST primary train/test video overlap: 0.
- T chronological boundaries are strict within every video.
- ST contains no ENV-03 frame in training.
- ST primary test contains only later ENV-03 frames.

No model training or model-performance evaluation is part of this phase.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--environment-map", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    environment_map_path = Path(args.environment_map)
    out = Path(args.output_dir)
    report_path = Path(args.report)

    flat, by_video, temporal_meta = load_inputs(manifest_path, environment_map_path)
    regimes = build_regimes(flat)
    master_ids = {str(r["sample_id"]) for r in flat}
    regime_summaries = audit_regimes(regimes, master_ids)

    out.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for name in ("O", "S", "T", "ST"):
        path = out / f"{name}.csv"
        write_csv(path, regimes[name])
        manifest_hashes[name] = sha256_file(path)

    summary = {
        "protocol_version": "v2-train-test-only",
        "source_manifest": {
            "path": manifest_path.as_posix(),
            "rows": len(flat),
            "sha256": sha256_file(manifest_path),
        },
        "environment_map": {
            "path": environment_map_path.as_posix(),
            "heldout_environment_for_S_and_ST": HELDOUT_ENVIRONMENT,
            "rows": len({r["video_id"] for r in flat}),
        },
        "temporal_rule": {
            "ordering": "ascending FrameNumber within canonical video_id",
            "cut": "floor(0.80 * observed_video_image_count)",
            "earlier": "rank < cut",
            "later": "rank >= cut",
            "all_videos_valid": True,
            "video_count": len(by_video),
            "min_video_length": min(v["length"] for v in temporal_meta.values()),
            "max_video_length": max(v["length"] for v in temporal_meta.values()),
        },
        "regime_manifest_sha256": manifest_hashes,
        "regimes": regime_summaries,
        "model_training_started": False,
        "model_test_predictions_examined": False,
    }
    (out / "SPLIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
