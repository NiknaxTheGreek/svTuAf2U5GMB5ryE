from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.splits import build_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen MonReader split manifests and registrations.")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/datasets/DATA-001.csv"))
    parser.add_argument("--environment-map", type=Path, default=Path("manifests/video_environment_map.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_split_artifacts(
        manifest_path=args.manifest,
        environment_map_path=args.environment_map,
        output_root=args.output_root,
        seed=args.seed,
    )
    compact = {
        split_id: {"name": reg["name"], "partition_counts": reg["summary"]["partition_counts"]}
        for split_id, reg in result["splits"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
