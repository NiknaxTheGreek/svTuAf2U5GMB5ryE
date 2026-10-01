from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.scratch_v2 import (
    EXPECTED_BANK_SHA256,
    EXPECTED_SPLIT_SHA256,
    ScratchCNN,
    ScratchCNNConfig,
    candidate_config,
    count_trainable_parameters,
    load_candidate_bank,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bank = load_candidate_bank(args.bank)
    records = []
    for i in range(1, 21):
        cid = f"C{i:02d}"
        pt = args.checkpoint_root / f"{cid}.pt"
        meta_path = args.checkpoint_root / f"{cid}.json"
        if not pt.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing frozen checkpoint files for {cid}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = candidate_config(bank, cid)
        if meta["candidate_config"] != config:
            raise ValueError(f"{cid} metadata config differs from frozen bank")
        if meta["checkpoint_sha256"] != sha256_file(pt):
            raise ValueError(f"{cid} checkpoint SHA-256 mismatch")
        if meta["status"] != "FROZEN_FINAL_EPOCH_20" or meta["epochs_completed"] != 20:
            raise ValueError(f"{cid} is not a final epoch-20 checkpoint")
        if meta["test_rows_loaded"] != 0 or meta["validation_rows_loaded"] != 0:
            raise ValueError(f"{cid} training touched test or validation data")
        if meta["bank_sha256"] != EXPECTED_BANK_SHA256:
            raise ValueError(f"{cid} bank identity mismatch")
        if meta["split_sha256"] != EXPECTED_SPLIT_SHA256["ST"]:
            raise ValueError(f"{cid} ST split identity mismatch")

        if meta.get("regime") != "ST":
            raise ValueError(f"{cid} metadata regime is not ST")
        checkpoint = torch.load(pt, map_location="cpu", weights_only=False)
        if checkpoint["candidate_id"] != cid or checkpoint["epoch"] != 20 or checkpoint.get("regime") != "ST":
            raise ValueError(f"{cid} checkpoint metadata mismatch")
        model = ScratchCNN(
            ScratchCNNConfig(
                depth=int(config["depth"]),
                start_filters=int(config["start_filters"]),
                dropout=float(config["dropout"]),
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        parameters = count_trainable_parameters(model)
        if parameters != int(meta["parameter_count"]):
            raise ValueError(f"{cid} parameter count mismatch")
        records.append({
            "candidate_id": cid,
            "checkpoint_file": pt.name,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "parameter_count": parameters,
            "candidate_config": config,
            "epochs_completed": 20,
            "final_train_loss": meta["final_train_loss"],
        })

    if len({row["checkpoint_sha256"] for row in records}) != 20:
        raise ValueError("Checkpoint SHA-256 values are not all unique")

    payload = {
        "status": "PASS_ALL_20_FROZEN_BEFORE_ST_TEST",
        "regime": "ST",
        "candidate_count": len(records),
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256["ST"],
        "test_evaluation_opened": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": payload["candidate_count"],
        "test_evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
