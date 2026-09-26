#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
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


def split_stats(rows: list[dict[str, object]], split_field: str, split_value: str) -> dict[str, int]:
    subset = [r for r in rows if r[split_field] == split_value]
    return {
        "frames": len(subset),
        "flip_frames": sum(r["label"] == "flip" for r in subset),
        "notflip_frames": sum(r["label"] == "notflip" for r in subset),
        "temporal_clips": len({r["sequence_id"] for r in subset}),
        "environment_groups": len({r["environment_group_id"] for r in subset}),
    }


def overlap_count(rows: list[dict[str, object]], split_field: str, a: str, b: str, key: str) -> int:
    left = {r[key] for r in rows if r[split_field] == a}
    right = {r[key] for r in rows if r[split_field] == b}
    return len(left & right)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-frame-manifest", required=True, type=Path)
    ap.add_argument("--dedup-membership", required=True, type=Path)
    ap.add_argument("--dedup-groups", required=True, type=Path)
    ap.add_argument("--dedup-summary", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_csv(args.d2_frame_manifest)
    membership = read_csv(args.dedup_membership)
    groups = read_csv(args.dedup_groups)
    dedup_summary = json.loads(args.dedup_summary.read_text(encoding="utf-8"))

    frame_required = {
        "sequence_id", "label", "video_id", "sequence_position", "frame_num",
        "supplied_split", "path", "sha256", "environment_group_id", "d2_split",
    }
    if not frames or not frame_required.issubset(frames[0]):
        raise SystemExit("D2 frame manifest schema mismatch")
    if len(frames) != 2989 or len({r["path"] for r in frames}) != len(frames):
        raise SystemExit("D2 frame manifest must contain 2,989 unique paths")

    membership_required = {
        "dedup_group_id", "keep", "path", "label", "video_id",
        "sequence_id", "frame_num", "split", "sha256",
    }
    if not membership or not membership_required.issubset(membership[0]):
        raise SystemExit("Dedup membership schema mismatch")
    if len({r["path"] for r in membership}) != len(membership):
        raise SystemExit("Duplicate paths in dedup membership")

    frame_by_path = {r["path"]: r for r in frames}
    unknown = sorted({r["path"] for r in membership} - set(frame_by_path))
    if unknown:
        raise SystemExit(f"Dedup membership contains unknown paths: {unknown[:5]}")

    # Validate the frozen dedup evidence before consuming it.
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in membership:
        by_group[r["dedup_group_id"]].append(r)

    if len(by_group) != len(groups):
        raise SystemExit("Dedup group count mismatch between membership and group summary")

    for gid, members in by_group.items():
        keepers = [r for r in members if r["keep"] == "1"]
        if len(keepers) != 1:
            raise SystemExit(f"Dedup group {gid} does not have exactly one representative")
        if len({r["sequence_id"] for r in members}) != 1:
            raise SystemExit(f"Dedup group {gid} spans multiple temporal clips")
        if len({r["label"] for r in members}) != 1:
            raise SystemExit(f"Dedup group {gid} spans multiple labels")
        for r in members:
            f = frame_by_path[r["path"]]
            if (
                r["sequence_id"] != f["sequence_id"]
                or r["label"] != f["label"]
                or r["split"] != f["supplied_split"]
                or r["sha256"] != f["sha256"]
            ):
                raise SystemExit(f"Dedup membership disagrees with frame manifest at {r['path']}")

    remove_paths = {r["path"] for r in membership if r["keep"] == "0"}
    representative_paths = {r["path"] for r in membership if r["keep"] == "1"}
    if remove_paths & representative_paths:
        raise SystemExit("A path is both removed and representative")

    expected_removed = int(dedup_summary["images_removed_if_one_rep_per_group"])
    expected_retained = int(dedup_summary["images_retained_after_dedup"])
    if len(remove_paths) != expected_removed:
        raise SystemExit("Removed-frame count disagrees with frozen dedup summary")
    if len(frames) - len(remove_paths) != expected_retained:
        raise SystemExit("Retained-frame count disagrees with frozen dedup summary")

    membership_by_path = {r["path"]: r for r in membership}
    audit_rows: list[dict[str, object]] = []
    for r in frames:
        m = membership_by_path.get(r["path"])
        if m is None:
            role = "unique"
            keep = 1
            gid = ""
        else:
            keep = int(m["keep"])
            role = "representative" if keep else "removed"
            gid = m["dedup_group_id"]
        audit_rows.append({
            **r,
            "dedup_group_id": gid,
            "dedup_role": role,
            "dedup_keep": keep,
        })

    audit_fields = list(frames[0].keys()) + ["dedup_group_id", "dedup_role", "dedup_keep"]
    audit_path = args.out_dir / "dedup_frame_audit.csv"
    write_csv(audit_path, audit_rows, audit_fields)

    kept = [r for r in audit_rows if r["dedup_keep"] == 1]
    removed = [r for r in audit_rows if r["dedup_keep"] == 0]

    d3_rows: list[dict[str, object]] = []
    for r in kept:
        out = dict(r)
        out["d3_split"] = out["supplied_split"]
        d3_rows.append(out)
    d3_fields = audit_fields + ["d3_split"]
    d3_path = args.out_dir / "d3_frame_manifest.csv"
    write_csv(d3_path, d3_rows, d3_fields)

    d4_rows: list[dict[str, object]] = []
    for r in kept:
        out = dict(r)
        out["d4_split"] = out["d2_split"]
        d4_rows.append(out)
    d4_fields = audit_fields + ["d4_split"]
    d4_path = args.out_dir / "d4_frame_manifest.csv"
    write_csv(d4_path, d4_rows, d4_fields)

    # D3 deliberately preserves the supplied split and is a repaired sensitivity
    # dataset, not a new claim of source independence.
    d3_train = split_stats(d3_rows, "d3_split", "training")
    d3_test = split_stats(d3_rows, "d3_split", "testing")
    d3_overlap = {
        "environment_group_overlap": overlap_count(
            d3_rows, "d3_split", "training", "testing", "environment_group_id"
        ),
        "sequence_overlap": overlap_count(
            d3_rows, "d3_split", "training", "testing", "sequence_id"
        ),
        "frame_path_overlap": overlap_count(
            d3_rows, "d3_split", "training", "testing", "path"
        ),
    }

    # D4 must inherit D2 exactly. Deduplication is a mask, not a new split search.
    d4_train = split_stats(d4_rows, "d4_split", "train")
    d4_test = split_stats(d4_rows, "d4_split", "test")
    d4_overlap = {
        "environment_group_overlap": overlap_count(
            d4_rows, "d4_split", "train", "test", "environment_group_id"
        ),
        "sequence_overlap": overlap_count(
            d4_rows, "d4_split", "train", "test", "sequence_id"
        ),
        "frame_path_overlap": overlap_count(
            d4_rows, "d4_split", "train", "test", "path"
        ),
    }
    if any(d4_overlap.values()):
        raise SystemExit(f"D4 violates frozen D2 isolation: {d4_overlap}")

    for r in d4_rows:
        if r["d4_split"] != r["d2_split"]:
            raise SystemExit(f"D4 assignment changed from D2 at {r['path']}")

    # Both D3 and D4 must be the same globally deduplicated population; only
    # assignment differs.
    d3_paths = {r["path"] for r in d3_rows}
    d4_paths = {r["path"] for r in d4_rows}
    if d3_paths != d4_paths or len(d3_paths) != expected_retained:
        raise SystemExit("D3/D4 retained populations disagree")

    retained_by_label = Counter(r["label"] for r in kept)
    removed_by_label = Counter(r["label"] for r in removed)
    removed_by_supplied = Counter(r["supplied_split"] for r in removed)
    removed_by_d2 = Counter(r["d2_split"] for r in removed)

    cross_supplied_group_count = sum(int(r["split_count"]) == 2 for r in groups)
    cross_supplied_rep_split = Counter(
        r["representative_split"] for r in groups if int(r["split_count"]) == 2
    )

    summary = {
        "dataset": "MonReader",
        "dedup_evidence": {
            "screen_rule": dedup_summary["screen_rule"],
            "confirm_rule": dedup_summary["confirm_rule"],
            "ssim_resolution": dedup_summary["ssim_resolution"],
            "near_duplicate_groups": len(groups),
            "images_in_near_duplicate_groups": len(membership),
            "removed_frames": len(remove_paths),
            "retained_frames": len(kept),
            "representative_rule": dedup_summary["representative_rule"],
            "groups_spanning_multiple_sequences": int(
                dedup_summary["groups_spanning_multiple_sequences"]
            ),
            "groups_spanning_both_labels": int(
                dedup_summary["groups_spanning_both_labels"]
            ),
            "groups_spanning_both_supplied_splits": cross_supplied_group_count,
            "cross_supplied_group_representative_split_counts": dict(
                sorted(cross_supplied_rep_split.items())
            ),
        },
        "population_effect": {
            "retained_by_label": dict(sorted(retained_by_label.items())),
            "removed_by_label": dict(sorted(removed_by_label.items())),
            "removed_by_supplied_split": dict(sorted(removed_by_supplied.items())),
            "removed_by_d2_split": dict(sorted(removed_by_d2.items())),
        },
        "d3": {
            "definition": "supplied_train_test_deduplicated",
            "status": "FROZEN",
            "train": d3_train,
            "test": d3_test,
            "overlap_checks": d3_overlap,
            "interpretation": (
                "Post-hoc deduplicated sensitivity dataset preserving the supplied "
                "train/test labels. Deduplication removes repeated frames but does not "
                "make the supplied split source-independent."
            ),
            "holdout_status": "NOT_UNTOUCHED_POST_HOC_REPAIR",
        },
        "d4": {
            "definition": "source_safe_deduplicated",
            "status": "FROZEN",
            "train": d4_train,
            "test": d4_test,
            "overlap_checks": d4_overlap,
            "split_reuse_check": "PASS",
            "interpretation": (
                "Uses the exact frozen D2 environment-group assignment, then applies "
                "the frozen global deduplication mask. No source split is re-optimized "
                "after deduplication."
            ),
            "holdout_policy": (
                "The D4 test partition inherits D2 protection and must not be used for "
                "model selection, threshold selection, debugging, or redesign."
            ),
        },
    }
    summary_path = args.out_dir / "d3_d4_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "inputs": {
            str(args.d2_frame_manifest): sha256(args.d2_frame_manifest),
            str(args.dedup_membership): sha256(args.dedup_membership),
            str(args.dedup_groups): sha256(args.dedup_groups),
            str(args.dedup_summary): sha256(args.dedup_summary),
        },
        "outputs": {
            audit_path.name: sha256(audit_path),
            d3_path.name: sha256(d3_path),
            d4_path.name: sha256(d4_path),
            summary_path.name: sha256(summary_path),
        },
        "acceptance": {
            "dedup_groups_have_exactly_one_representative": True,
            "dedup_groups_single_sequence": True,
            "dedup_groups_single_label": True,
            "d3_d4_same_retained_population": True,
            "d4_reuses_d2_assignment": True,
            "d4_zero_environment_overlap": d4_overlap["environment_group_overlap"] == 0,
            "d4_zero_sequence_overlap": d4_overlap["sequence_overlap"] == 0,
            "d4_zero_frame_path_overlap": d4_overlap["frame_path_overlap"] == 0,
        },
        "status": "PASS",
    }
    receipt_path = args.out_dir / "d3_d4_split_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
