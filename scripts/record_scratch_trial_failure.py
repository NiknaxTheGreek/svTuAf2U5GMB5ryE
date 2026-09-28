from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import validate_stage1_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-config", required=True, type=Path)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    payload = json.loads(args.trial_config.read_text(encoding="utf-8"))
    validate_stage1_payload(payload)
    trial = next(item for item in payload["trials"] if item["trial_id"] == args.trial_id)
    failure_type = "timeout" if args.exit_code in {124, 137, 143} else "external_process_failure"
    record = {
        "trial_id": args.trial_id,
        "stage": "random",
        "status": "failed",
        "config": trial,
        "device": "cpu",
        "failure_type": failure_type,
        "failure_message": f"trial process exited with code {args.exit_code}",
        "exit_code": args.exit_code,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
