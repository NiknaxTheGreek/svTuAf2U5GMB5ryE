from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import select_top_combined_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the five candidates entering the source-disjoint gate.")
    parser.add_argument("--stage1-results", required=True, type=Path)
    parser.add_argument("--stage2-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stage1 = json.loads(args.stage1_results.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2_results.read_text(encoding="utf-8"))
    top5 = select_top_combined_candidates(stage1["results"], stage2["results"], count=5)
    payload = {
        "selection_metric": "original_tuning_validation_f1",
        "count": 5,
        "candidate_ids": [record["trial_id"] for record in top5],
        "candidates": [
            {
                "trial_id": record["trial_id"],
                "best_validation_f1": record["best_validation_f1"],
                "config": record["config"],
            }
            for record in top5
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_ids": payload["candidate_ids"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
