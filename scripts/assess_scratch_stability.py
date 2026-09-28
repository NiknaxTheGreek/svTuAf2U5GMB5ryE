from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import assess_three_seed_stability


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the locked three-seed scratch-CNN stability gate.")
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.results_root.rglob("stability-seed-*.json"))
    ]
    if len(records) != 3:
        raise ValueError("Exactly three stability result files are required")
    source_ids = {record.get("source_candidate_id") for record in records}
    if source_ids != {canonical["source_candidate_id"]}:
        raise ValueError("Stability runs do not all correspond to the canonical config")
    normalized = [
        {
            "seed": int(record["seed"]),
            "status": record["status"],
            "validation_f1": float(record["best_validation_f1"]),
        }
        for record in records
    ]
    assessment = assess_three_seed_stability(normalized)
    payload = {
        "source_candidate_id": canonical["source_candidate_id"],
        "config": canonical["config"],
        "assessment": assessment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(assessment, sort_keys=True))
    return 0 if assessment["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
