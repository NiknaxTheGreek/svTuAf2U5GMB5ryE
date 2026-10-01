from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.scratch_v2 import EXPECTED_SPLIT_SHA256, read_csv, sha256_file

SEED = "monreader-inpainting-poc-v2"
TARGET = 100


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{sample_id}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-summary", required=True, type=Path)
    args = ap.parse_args()

    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("ST split hash mismatch")

    rows = read_csv(args.membership)
    context = [r for r in rows if r["role"] == "context"]
    primary_test = {r["sample_id"] for r in rows if r["role"] == "test"}
    if len(context) != 1072:
        raise ValueError(f"Expected 1072 ST context rows, got {len(context)}")

    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in context:
        strata[(r["role_detail"], r["environment_id"], r["label"])].append(r)

    keys = sorted(strata)
    if not keys:
        raise ValueError("No non-empty context strata")

    base = TARGET // len(keys)
    remainder = TARGET % len(keys)
    allocations = {
        key: base + (1 if i < remainder else 0)
        for i, key in enumerate(keys)
    }

    selected = []
    stratum_summary = []
    for key in keys:
        bucket = sorted(strata[key], key=lambda r: stable_key(r["sample_id"]))
        n = allocations[key]
        if len(bucket) < n:
            raise ValueError(f"Stratum {key} has {len(bucket)} rows but needs {n}")
        chosen = bucket[:n]
        selected.extend(chosen)
        stratum_summary.append({
            "role_detail": key[0],
            "environment_id": key[1],
            "label": key[2],
            "available": len(bucket),
            "selected": len(chosen),
        })

    selected = sorted(
        selected,
        key=lambda r: (r["role_detail"], r["environment_id"], r["label"], stable_key(r["sample_id"])),
    )
    if len(selected) != TARGET:
        raise RuntimeError(f"Expected {TARGET} selected rows, got {len(selected)}")
    if len({r["sample_id"] for r in selected}) != TARGET:
        raise RuntimeError("Duplicate sample ID in PoC set")
    overlap = {r["sample_id"] for r in selected} & primary_test
    if overlap:
        raise RuntimeError(f"Primary-test overlap detected: {sorted(overlap)[:5]}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id","role_detail","environment_id","label","video_id",
        "frame_number","supplied_split"
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({k:r[k] for k in fields})

    summary = {
        "status": "FROZEN_CONTEXT_ONLY_POC_SAMPLE",
        "seed": SEED,
        "target_images": TARGET,
        "selected_images": len(selected),
        "context_population": len(context),
        "primary_test_overlap": 0,
        "strata": stratum_summary,
        "sample_csv_sha256": sha256_file(args.output_csv),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
