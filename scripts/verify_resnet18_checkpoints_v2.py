from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.resnet18_v2 import (
    EXPECTED_CONFIG_SHA256,
    build_model,
    load_fixed_config,
    total_parameter_count,
)
from src.scratch_v2 import EXPECTED_SPLIT_SHA256, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_fixed_config(args.config)
    records = []
    for regime in ("O","S","T","ST"):
        pt = args.checkpoint_root / f"{regime}.pt"
        meta_path = args.checkpoint_root / f"{regime}.json"
        if not pt.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing ResNet18 frozen checkpoint for {regime}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["status"] != "FROZEN_FINAL_EPOCH_20" or meta["epochs_completed"] != 20:
            raise ValueError(f"{regime} checkpoint is not final epoch 20")
        if meta["head_only_epochs"] != 5 or meta["full_network_epochs"] != 15:
            raise ValueError(f"{regime} phase schedule changed")
        if meta["test_rows_loaded"] != 0 or meta["validation_rows_loaded"] != 0:
            raise ValueError(f"{regime} training touched test/validation")
        if meta["config_sha256"] != EXPECTED_CONFIG_SHA256:
            raise ValueError(f"{regime} comparator config identity mismatch")
        if meta["split_sha256"] != EXPECTED_SPLIT_SHA256[regime]:
            raise ValueError(f"{regime} split identity mismatch")
        if meta["checkpoint_sha256"] != sha256_file(pt):
            raise ValueError(f"{regime} checkpoint SHA mismatch")
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        if payload["regime"] != regime or payload["epoch"] != 20:
            raise ValueError(f"{regime} checkpoint payload mismatch")
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if total_parameter_count(model) != int(meta["total_parameter_count"]):
            raise ValueError(f"{regime} parameter count mismatch")
        records.append({
            "regime": regime,
            "checkpoint_file": pt.name,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "train_rows": meta["train_rows"],
            "final_train_loss": meta["final_train_loss"],
            "elapsed_seconds": meta["elapsed_seconds"],
            "total_parameter_count": meta["total_parameter_count"],
        })

    if len({r["checkpoint_sha256"] for r in records}) != 4:
        raise ValueError("ResNet18 checkpoint hashes are not all unique")

    payload = {
        "status": "PASS_ALL_4_FROZEN_BEFORE_RESNET_TESTS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "regimes": ["O","S","T","ST"],
        "checkpoint_count": 4,
        "test_evaluation_opened": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "checkpoint_count": 4,
        "test_evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
