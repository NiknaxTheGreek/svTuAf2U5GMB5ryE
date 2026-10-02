from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SEED = "monreader-chroma-validation-v2"
TARGET = 100
EXPECTED_ST_SPLIT_SHA256 = "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{sample_id}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-membership", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-summary", type=Path, required=True)
    args = ap.parse_args()

    if sha256_file(args.st_membership) != EXPECTED_ST_SPLIT_SHA256:
        raise ValueError("ST split hash mismatch")

    rows = read_csv(args.st_membership)
    context = [r for r in rows if r["role"] == "context"]
    test_ids = {r["sample_id"] for r in rows if r["role"] == "test"}
    excluded = {r["sample_id"] for r in read_csv(args.exclude)}

    eligible = [r for r in context if r["sample_id"] not in excluded]
    if len(eligible) < TARGET:
        raise ValueError("Not enough eligible context rows")

    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in eligible:
        strata[(r["role_detail"], r["environment_id"], r["label"])].append(r)
    for key in strata:
        strata[key].sort(key=lambda r: stable_key(r["sample_id"]))

    keys = sorted(strata)
    positions = {key: 0 for key in keys}
    selected: list[dict[str, str]] = []

    # Deterministic round-robin across non-empty strata until 100 are selected.
    while len(selected) < TARGET:
        progressed = False
        for key in keys:
            pos = positions[key]
            bucket = strata[key]
            if pos < len(bucket) and len(selected) < TARGET:
                selected.append(bucket[pos])
                positions[key] = pos + 1
                progressed = True
        if not progressed:
            break

    if len(selected) != TARGET:
        raise RuntimeError(f"Selected {len(selected)} rows, expected {TARGET}")
    ids = [r["sample_id"] for r in selected]
    if len(set(ids)) != TARGET:
        raise RuntimeError("Duplicate validation sample IDs")
    if set(ids) & excluded:
        raise RuntimeError("Development/validation overlap")
    if set(ids) & test_ids:
        raise RuntimeError("Primary-test overlap")

    fields = [
        "sample_id","role_detail","environment_id","label",
        "video_id","frame_number","supplied_split"
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({k: r[k] for k in fields})

    counts = defaultdict(int)
    for r in selected:
        counts[(r["role_detail"], r["environment_id"], r["label"])] += 1

    summary = {
        "status": "FROZEN_CHROMA_VALIDATION_100",
        "seed": SEED,
        "selected_images": TARGET,
        "development_overlap": 0,
        "primary_test_overlap": 0,
        "remaining_context_population_before_selection": len(eligible),
        "sample_csv_sha256": sha256_file(args.output_csv),
        "strata_counts": [
            {
                "role_detail": k[0],
                "environment_id": k[1],
                "label": k[2],
                "selected": counts[k],
            }
            for k in sorted(counts)
        ],
    }
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
