from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import stage1_payload, validate_stage1_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the frozen Stage-1 scratch-CNN random search.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/sweeps/scratch_stage1_random.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = stage1_payload()
    validate_stage1_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"trials={len(payload['trials'])}")
    print(f"zero_weight_decay={sum(float(t['weight_decay']) == 0.0 for t in payload['trials'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
