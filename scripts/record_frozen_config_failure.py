from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_scratch_config import load_frozen_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an external failure for one frozen scratch config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--failure-type", default=None)
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()

    config = load_frozen_config(args.config)
    if args.failure_type is not None and not args.evidence:
        raise ValueError("Reviewed failure classification requires evidence")
    failure_type = args.failure_type
    if failure_type is None:
        if args.exit_code == 124:
            failure_type = "timeout"
        elif args.exit_code in {137, 143}:
            failure_type = "process_terminated"
        else:
            failure_type = "external_process_failure"
    record: dict[str, object] = {
        "trial_id": str(config["trial_id"]),
        "stage": args.stage,
        "status": "failed",
        "config": config,
        "device": "cpu",
        "failure_type": failure_type,
        "failure_message": f"trial process exited with code {args.exit_code}",
        "exit_code": args.exit_code,
    }
    if "source_candidate_id" in config:
        record["source_candidate_id"] = str(config["source_candidate_id"])
    if args.evidence:
        record["failure_evidence"] = args.evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
