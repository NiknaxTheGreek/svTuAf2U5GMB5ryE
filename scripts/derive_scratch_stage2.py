from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tuning import STAGE1_TRIAL_COUNT, STAGE2_TRIAL_COUNT, derive_stage2_search_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the locked Stage-2 scratch-CNN search envelope from the Stage-1 top eight."
    )
    parser.add_argument("--stage1-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage1 = json.loads(args.stage1_results.read_text(encoding="utf-8"))
    if stage1.get("stage") != "random" or stage1.get("trial_count") != STAGE1_TRIAL_COUNT:
        raise ValueError("Input is not the frozen reduced Stage-1 aggregate")
    results = stage1.get("results")
    if not isinstance(results, list) or len(results) != STAGE1_TRIAL_COUNT:
        raise ValueError(f"Stage-1 aggregate must account for exactly {STAGE1_TRIAL_COUNT} trials")
    by_id = {record["trial_id"]: record for record in results}
    top8_ids = stage1.get("top8_trial_ids")
    if not isinstance(top8_ids, list) or len(top8_ids) != 8:
        raise ValueError("Stage-1 aggregate does not contain exactly eight ranked trial IDs")
    top8 = [by_id[trial_id] for trial_id in top8_ids]
    space = derive_stage2_search_space(top8)
    payload = {
        "stage": "bayesian",
        "seed": 42,
        "trial_count": STAGE2_TRIAL_COUNT,
        "source_stage1_top8": top8_ids,
        "search_space": space,
        "fixed": {
            "augmentation": False,
            "canvas": {"height": 398, "width": 224},
            "class_sampling": "natural",
            "early_stopping_patience": 4,
            "gradient_clipping": None,
            "loss": "BCEWithLogitsLoss",
            "max_epochs": 20,
            "scaling": "[0,1]",
            "scheduler": None,
            "seed": 42,
            "threshold": 0.5,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"top8_trial_ids": top8_ids, "search_space": space}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
