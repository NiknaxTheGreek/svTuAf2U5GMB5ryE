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

from src.resnet18_v2 import (
    EXPECTED_CONFIG_SHA256,
    ResNetMemmapDataset,
    build_model,
    load_fixed_config,
    set_head_only,
    total_parameter_count,
    trainable_parameter_count,
    unfreeze_all,
    verify_cache_receipt,
)
from src.scratch_v2 import EXPECTED_SPLIT_SHA256, make_generator, seed_everything, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-data", required=True, type=Path)
    ap.add_argument("--cache-index", required=True, type=Path)
    ap.add_argument("--cache-receipt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--regime", required=True, choices=("O","S","T","ST"))
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_fixed_config(args.config)
    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    verify_cache_receipt(receipt, args.regime, "train", args.cache_data, args.cache_index)

    seed_everything(int(cfg["seed"]))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    dataset = ResNetMemmapDataset(args.cache_data, args.cache_index)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=make_generator(int(cfg["seed"])),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = build_model(cfg, pretrained=True)
    set_head_only(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["head_learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    history = []
    started = time.perf_counter()
    for epoch in range(1, 21):
        if epoch == 6:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["finetune_learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
            )
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"]).squeeze(1)
            labels = batch["label"]
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            n = labels.shape[0]
            total_loss += float(loss.detach()) * n
            total_n += n
        mean_loss = total_loss / total_n
        phase = "head_only" if epoch <= 5 else "full_network"
        history.append({"epoch": epoch, "phase": phase, "train_loss": mean_loss})
        print(f"regime={args.regime} epoch={epoch:02d}/20 phase={phase} train_loss={mean_loss:.8f}", flush=True)

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt = args.output_dir / f"{args.regime}.pt"
    meta_path = args.output_dir / f"{args.regime}.json"
    payload = {
        "protocol_version": "v2-train-test-only",
        "model_family": "resnet18_fixed_imagenet_comparator",
        "regime": args.regime,
        "epoch": 20,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, pt)
    checkpoint_sha = sha256_file(pt)

    reloaded = torch.load(pt, map_location="cpu", weights_only=False)
    fresh = build_model(cfg, pretrained=False)
    fresh.load_state_dict(reloaded["model_state_dict"], strict=True)

    metadata = {
        "status": "FROZEN_FINAL_EPOCH_20",
        "model_family": "resnet18_fixed_imagenet_comparator",
        "regime": args.regime,
        "epochs_completed": 20,
        "head_only_epochs": 5,
        "full_network_epochs": 15,
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "checkpoint_file": pt.name,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "cache_data_sha256": receipt["cache_data_sha256"],
        "cache_index_sha256": receipt["cache_index_sha256"],
        "train_rows": len(dataset),
        "test_rows_loaded": 0,
        "validation_rows_loaded": 0,
        "threshold": 0.5,
        "total_parameter_count": total_parameter_count(model),
        "final_trainable_parameter_count": trainable_parameter_count(model),
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
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "regime": args.regime,
        "checkpoint_sha256": checkpoint_sha,
        "epochs_completed": 20,
        "test_rows_loaded": 0,
        "validation_rows_loaded": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
