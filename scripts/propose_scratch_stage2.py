from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import propose_stage2_bayesian_trial, scratch_parameter_count


def load_prior_records(root: Path) -> list[dict[str, object]]:
    records = []
    if root.exists():
        for path in sorted(root.rglob("stage2-*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    ids = [record["trial_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Stage-2 result records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Propose one deterministic Stage-2 Bayesian trial.")
    parser.add_argument("--stage1-results", required=True, type=Path)
    parser.add_argument("--stage2-space", required=True, type=Path)
    parser.add_argument("--prior-results-root", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not 1 <= args.iteration <= 30:
        raise ValueError("Stage-2 iteration must be in [1,30]")
    stage1 = json.loads(args.stage1_results.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2_space.read_text(encoding="utf-8"))
    if stage1.get("stage") != "random" or stage1.get("trial_count") != 40:
        raise ValueError("Invalid Stage-1 aggregate")
    if stage2.get("stage") != "bayesian" or stage2.get("trial_count") != 30:
        raise ValueError("Invalid Stage-2 search-space registration")

    by_id = {record["trial_id"]: record for record in stage1["results"]}
    top8_ids = list(stage2["source_stage1_top8"])
    if top8_ids != list(stage1["top8_trial_ids"]):
        raise ValueError("Stage-2 source top-eight does not match Stage-1 aggregate")
    observations = [by_id[trial_id] for trial_id in top8_ids]
    prior = load_prior_records(args.prior_results_root)
    expected_prior_ids = [f"stage2-{index:03d}" for index in range(1, args.iteration)]
    observed_prior_ids = sorted(record["trial_id"] for record in prior)
    if observed_prior_ids != expected_prior_ids:
        raise ValueError(
            f"Stage-2 prior result sequence is incomplete: {observed_prior_ids} != {expected_prior_ids}"
        )
    observations.extend(prior)
    proposal = propose_stage2_bayesian_trial(
        stage2["search_space"],
        observations,
        iteration=args.iteration,
        seed=int(stage2.get("seed", 42)),
    )
    proposal["parameter_count"] = scratch_parameter_count(
        int(proposal["depth"]), int(proposal["start_filters"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(proposal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
