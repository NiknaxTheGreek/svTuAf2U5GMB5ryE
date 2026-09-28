from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import scratch_parameter_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the five source-disjoint gate configs.")
    parser.add_argument("--top5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    top5 = json.loads(args.top5.read_text(encoding="utf-8"))
    candidates = list(top5["candidates"])
    if len(candidates) != 5:
        raise ValueError("Source-disjoint gate requires exactly five frozen candidates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, candidate in enumerate(candidates, start=1):
        source_id = str(candidate["trial_id"])
        config = dict(candidate["config"])
        config.update(
            {
                "trial_id": f"source-gate-{rank:02d}",
                "source_candidate_id": source_id,
                "seed": 42,
                "parameter_count": scratch_parameter_count(
                    int(config["depth"]), int(config["start_filters"])
                ),
            }
        )
        path = args.output_dir / f"source-gate-{rank:02d}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append({"rank": rank, "source_candidate_id": source_id, "path": path.name})
    print(json.dumps({"configs": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
