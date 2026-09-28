from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import validate_stage1_payload


ALLOWED_OVERRIDES = {
    "out_of_memory",
    "infrastructure_failure",
    "timeout",
    "external_process_failure",
}


def _default_failure_type(exit_code: int) -> str:
    if exit_code == 124:
        return "timeout"
    if exit_code == 137:
        return "process_killed"
    if exit_code == 143:
        return "process_terminated"
    return "external_process_failure"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-config", required=True, type=Path)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument(
        "--failure-type",
        choices=sorted(ALLOWED_OVERRIDES),
        default=None,
        help="Optional reviewed classification. Use only when external evidence supports it.",
    )
    parser.add_argument(
        "--evidence",
        default=None,
        help="Concise evidence supporting a reviewed failure classification.",
    )
    args = parser.parse_args()

    payload = json.loads(args.trial_config.read_text(encoding="utf-8"))
    validate_stage1_payload(payload)
    matches = [item for item in payload["trials"] if item["trial_id"] == args.trial_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen config for {args.trial_id}")
    trial = matches[0]

    failure_type = args.failure_type or _default_failure_type(args.exit_code)
    if args.failure_type is not None and not args.evidence:
        raise ValueError("Reviewed failure classification requires --evidence")

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
    if args.evidence:
        record["failure_evidence"] = args.evidence

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
