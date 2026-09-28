from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the canonical scratch config from source-disjoint gate results.")
    parser.add_argument("--top5", required=True, type=Path)
    parser.add_argument("--gate-results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    top5 = json.loads(args.top5.read_text(encoding="utf-8"))
    expected_ids = list(top5["candidate_ids"])
    records = []
    for path in sorted(args.gate_results_root.rglob("source-gate-*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    by_id = {str(record["source_candidate_id"]): record for record in records}
    if sorted(by_id) != sorted(expected_ids):
        raise ValueError("Source-disjoint gate results do not cover the frozen top five")
    if any(by_id[trial_id].get("status") != "success" for trial_id in expected_ids):
        raise ValueError("All five source-disjoint gate runs must succeed before canonical selection")
    ranked = sorted(
        (by_id[trial_id] for trial_id in expected_ids),
        key=lambda record: (
            -float(record["best_validation_f1"]),
            str(record["source_candidate_id"]),
        ),
    )
    winner = ranked[0]
    payload = {
        "selection_metric": "source_disjoint_validation_f1",
        "source_candidate_id": winner["source_candidate_id"],
        "best_validation_f1": winner["best_validation_f1"],
        "config": winner["config"],
        "ranking": [
            {
                "source_candidate_id": record["source_candidate_id"],
                "best_validation_f1": record["best_validation_f1"],
            }
            for record in ranked
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"source_candidate_id": payload["source_candidate_id"], "f1": payload["best_validation_f1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
