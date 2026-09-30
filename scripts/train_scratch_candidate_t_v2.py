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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-data", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--regime", default="T", choices=("T",))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    bank = load_candidate_bank(args.bank)
    config = candidate_config(bank, args.candidate_id)
    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    if receipt["regime"] != args.regime or receipt["role"] != "train":
        raise ValueError("Training cache is not the requested regime/train population")
    if receipt["split_sha256"] != EXPECTED_SPLIT_SHA256[args.regime]:
        raise ValueError("Training-cache split identity mismatch")
    if receipt["rows"] != 2365:
        raise ValueError(f"T training cache must contain 2365 rows, found {receipt['rows']}")
    if sha256_file(args.cache_data) != receipt["cache_data_sha256"]:
        raise ValueError("Training cache-data hash mismatch")
    if sha256_file(args.cache_index) != receipt["cache_index_sha256"]:
        raise ValueError("Training cache-index hash mismatch")

    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    dataset = MemmapDataset(args.cache_data, args.cache_index)
    if len(dataset) != 2365:
        raise ValueError("T train dataset length changed")

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
    device = torch.device("cpu")
    model.to(device)

    history = []
    started = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {args.candidate_id} at epoch {epoch}")
            loss.backward()
            optimizer.step()
            n = labels.shape[0]
            total_loss += float(loss.detach()) * n
            total_n += n
        mean_loss = total_loss / total_n
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"candidate={args.candidate_id} epoch={epoch:02d}/20 train_loss={mean_loss:.8f}", flush=True)

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.candidate_id}.pt"
    metadata_path = args.output_dir / f"{args.candidate_id}.json"

    payload = {
        "protocol_version": "v2-train-test-only",
        "regime": args.regime,
        "candidate_id": args.candidate_id,
        "epoch": 20,
        "candidate_config": config,
        "fixed_training": bank["fixed_training"],
        "optimizer_resolved": resolved_optimizer_settings(config),
        "parameter_count": parameter_count,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "cache_data_sha256": receipt["cache_data_sha256"],
        "cache_index_sha256": receipt["cache_index_sha256"],
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    # Mandatory round-trip validation before the checkpoint is accepted.
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fresh = ScratchCNN(
        ScratchCNNConfig(
            depth=int(config["depth"]),
            start_filters=int(config["start_filters"]),
            dropout=float(config["dropout"]),
        )
    )
    fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
    if int(reloaded["epoch"]) != 20:
        raise RuntimeError("Reloaded checkpoint is not epoch 20")
    if int(reloaded["parameter_count"]) != parameter_count:
        raise RuntimeError("Reloaded checkpoint parameter count changed")

    metadata = {
        "status": "FROZEN_FINAL_EPOCH_20",
        "regime": args.regime,
        "candidate_id": args.candidate_id,
        "candidate_config": config,
        "optimizer_resolved": resolved_optimizer_settings(config),
        "parameter_count": parameter_count,
        "train_rows": len(dataset),
        "epochs_completed": 20,
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "cache_data_sha256": receipt["cache_data_sha256"],
        "cache_index_sha256": receipt["cache_index_sha256"],
        "test_rows_loaded": 0,
        "validation_rows_loaded": 0,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_id": args.candidate_id,
        "checkpoint_sha256": checkpoint_sha,
        "parameter_count": parameter_count,
        "epochs_completed": 20,
        "test_rows_loaded": 0,
        "validation_rows_loaded": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
