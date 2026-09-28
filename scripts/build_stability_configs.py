from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import scratch_parameter_count


DEFAULT_STABILITY_SEEDS = (42, 43, 44)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the three canonical scratch stability configs.")
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    source_id = str(canonical["source_candidate_id"])
    base = dict(canonical["config"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in DEFAULT_STABILITY_SEEDS:
        config = dict(base)
        config.update(
            {
                "trial_id": f"stability-seed-{seed}",
                "source_candidate_id": source_id,
                "seed": seed,
                "parameter_count": scratch_parameter_count(
                    int(config["depth"]), int(config["start_filters"])
                ),
            }
        )
        path = args.output_dir / f"stability-seed-{seed}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append({"seed": seed, "path": path.name})
    registration = {
        "source_candidate_id": source_id,
        "seeds": list(DEFAULT_STABILITY_SEEDS),
        "selection": "implementation-locked before stability execution",
        "acceptance": {
            "f1_population_std_max": 0.02,
            "every_seed_within_best_f1": 0.02,
        },
    }
    (args.output_dir / "stability_protocol.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registration, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
