from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import validate_stage1_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard index")
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    validate_stage1_payload(payload)
    selected = [
        trial["trial_id"]
        for index, trial in enumerate(payload["trials"])
        if index % args.shards == args.shard
    ]
    for trial_id in selected:
        print(trial_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
