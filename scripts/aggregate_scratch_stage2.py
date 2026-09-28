from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the 30 sequential Stage-2 trials.")
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    by_id: dict[str, dict[str, object]] = {}
    for path in sorted(args.results_root.rglob("stage2-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        trial_id = str(record.get("trial_id"))
        if trial_id in by_id:
            raise ValueError(f"Duplicate Stage-2 result: {trial_id}")
        by_id[trial_id] = record
    expected = [f"stage2-{index:03d}" for index in range(1, 31)]
    if sorted(by_id) != expected:
        raise ValueError(f"Stage-2 results incomplete: found {sorted(by_id)}")
    records = [by_id[trial_id] for trial_id in expected]
    successes = [
        record
        for record in records
        if record.get("status") == "success"
    ]
    failures = [record for record in records if record.get("status") != "success"]
    payload = {
        "stage": "bayesian",
        "trial_count": 30,
        "success_count": len(successes),
        "failure_count": len(failures),
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"success_count": len(successes), "failure_count": len(failures)}, sort_keys=True))
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
