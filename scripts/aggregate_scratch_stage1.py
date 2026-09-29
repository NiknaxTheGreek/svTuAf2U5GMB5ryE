from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import STAGE1_TRIAL_COUNT, validate_stage1_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.trial_config.read_text(encoding="utf-8"))
    validate_stage1_payload(config)
    result_files = list(args.results_root.rglob("stage1-*.json"))
    by_id = {}
    for path in result_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        trial_id = record.get("trial_id")
        if trial_id in by_id:
            raise ValueError(f"Duplicate result for {trial_id}")
        by_id[trial_id] = record

    records = []
    for trial in config["trials"]:
        trial_id = trial["trial_id"]
        record = by_id.get(trial_id)
        if record is None:
            record = {
                "trial_id": trial_id,
                "stage": "random",
                "status": "failed",
                "config": trial,
                "failure_type": "missing_result",
                "failure_message": "No result artifact was produced by the shard.",
            }
        if record.get("config") != trial:
            raise ValueError(f"Executed config differs from frozen config for {trial_id}")
        records.append(record)

    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] != "success"]
    ranked = sorted(
        successes,
        key=lambda record: (-float(record["best_validation_f1"]), record["trial_id"]),
    )
    payload = {
        "stage": "random",
        "trial_count": STAGE1_TRIAL_COUNT,
        "success_count": len(successes),
        "failure_count": len(failures),
        "top8_trial_ids": [record["trial_id"] for record in ranked[:8]],
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "success_count": len(successes),
                "failure_count": len(failures),
                "top8_trial_ids": payload["top8_trial_ids"],
            },
            sort_keys=True,
        )
    )
    return 0 if len(successes) >= 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
