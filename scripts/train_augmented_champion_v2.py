from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.scratch_v2 import (
    EXPECTED_BANK_SHA256,
    EXPECTED_SPLIT_SHA256,
    MemmapDataset,
    ScratchCNN,
    ScratchCNNConfig,
    build_optimizer,
    candidate_config,
    count_trainable_parameters,
    load_candidate_bank,
    make_generator,
    resolved_optimizer_settings,
    seed_everything,
    sha256_file,
)
from src.secondary_augmentation_v2 import (
    EXPECTED_AUGMENTATION_CONFIG_SHA256,
    augment_batch,
    load_augmentation_config,
)

CHAMPIONS = {"O": "C14", "S": "C18", "T": "C06", "ST": "C07"}
TRAIN_ROWS = {"O": 2392, "S": 2219, "T": 2365, "ST": 1756}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-data", required=True, type=Path)
    ap.add_argument("--cache-index", required=True, type=Path)
    ap.add_argument("--cache-receipt", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--augmentation-config", required=True, type=Path)
    ap.add_argument("--regime", required=True, choices=("O","S","T","ST"))
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    bank = load_candidate_bank(args.bank)
    aug_cfg = load_augmentation_config(args.augmentation_config)
    candidate_id = CHAMPIONS[args.regime]
    if aug_cfg["champion_configs"][args.regime] != candidate_id:
        raise ValueError("Frozen augmentation champion map changed")
    config = candidate_config(bank, candidate_id)

    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    if receipt["regime"] != args.regime or receipt["role"] != "train":
        raise ValueError("Training cache population mismatch")
    if receipt["split_sha256"] != EXPECTED_SPLIT_SHA256[args.regime]:
        raise ValueError("Training cache split mismatch")
    if receipt["rows"] != TRAIN_ROWS[args.regime]:
        raise ValueError("Training cache row count changed")
    if sha256_file(args.cache_data) != receipt["cache_data_sha256"]:
        raise ValueError("Training cache data hash mismatch")
    if sha256_file(args.cache_index) != receipt["cache_index_sha256"]:
        raise ValueError("Training cache index hash mismatch")

    seed = int(config["seed"])
    if seed != 42:
        raise ValueError("Frozen champion seed changed")
    seed_everything(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    dataset = MemmapDataset(args.cache_data, args.cache_index)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = ScratchCNN(
        ScratchCNNConfig(
            depth=int(config["depth"]),
            start_filters=int(config["start_filters"]),
            dropout=float(config["dropout"]),
        )
    )
    parameter_count = count_trainable_parameters(model)
    optimizer = build_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    started = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            images = augment_batch(batch["image"], aug_cfg)
            labels = batch["label"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite augmented loss in {args.regime} epoch {epoch}")
            loss.backward()
            optimizer.step()
            n = labels.shape[0]
            total_loss += float(loss.detach()) * n
            total_n += n
        mean_loss = total_loss / total_n
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"regime={args.regime} candidate={candidate_id} augmented epoch={epoch:02d}/20 train_loss={mean_loss:.8f}", flush=True)

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt = args.output_dir / f"{args.regime}_{candidate_id}_aug.pt"
    meta_path = args.output_dir / f"{args.regime}_{candidate_id}_aug.json"
    payload = {
        "protocol_version": "v2-secondary-posthoc",
        "experiment": "fixed_mild_augmentation",
        "regime": args.regime,
        "candidate_id": candidate_id,
        "epoch": 20,
        "candidate_config": config,
        "optimizer_resolved": resolved_optimizer_settings(config),
        "parameter_count": parameter_count,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "augmentation_config_sha256": EXPECTED_AUGMENTATION_CONFIG_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, pt)
    checkpoint_sha = sha256_file(pt)

    reloaded = torch.load(pt, map_location="cpu", weights_only=False)
    fresh = ScratchCNN(
        ScratchCNNConfig(
            depth=int(config["depth"]),
            start_filters=int(config["start_filters"]),
            dropout=float(config["dropout"]),
        )
    )
    fresh.load_state_dict(reloaded["model_state_dict"], strict=True)

    meta = {
        "status": "FROZEN_AUGMENTED_FINAL_EPOCH_20",
        "selection_bearing": False,
        "regime": args.regime,
        "candidate_id": candidate_id,
        "candidate_config": config,
        "parameter_count": parameter_count,
        "epochs_completed": 20,
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "checkpoint_file": pt.name,
        "checkpoint_sha256": checkpoint_sha,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "augmentation_config_sha256": EXPECTED_AUGMENTATION_CONFIG_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "train_rows": len(dataset),
        "test_rows_loaded": 0,
        "validation_rows_loaded": 0,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "regime": args.regime, "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha,
        "test_rows_loaded": 0, "validation_rows_loaded": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
