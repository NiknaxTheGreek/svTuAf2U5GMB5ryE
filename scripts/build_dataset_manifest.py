from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import build_dataset_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical MonReader dataset manifest.")
    parser.add_argument("--archive", required=True, help="Path to the authoritative images.zip archive.")
    parser.add_argument("--output-root", required=True, help="Directory in which to create generated artifacts.")
    parser.add_argument("--representative-per-stratum", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_dataset_artifacts(
        Path(args.archive),
        Path(args.output_root),
        representative_per_stratum=args.representative_per_stratum,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
