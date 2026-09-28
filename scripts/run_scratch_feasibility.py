from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MonReaderDirectoryDataset, join_manifest_with_membership
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.preprocessing import build_base_transform
from src.training import build_optimizer, run_binary_training
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real MonReader scratch-CNN CPU feasibility epoch.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = 42
    seed_everything(seed)
    rows = join_manifest_with_membership(
        "manifests/datasets/DATA-001.csv",
        "manifests/splits/SPLIT-007_original_tuning.csv",
        allowed_partitions={"fit", "validation"},
    )
    train_rows = [row for row in rows if row["partition"] == "fit"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    transform = build_base_transform("manifests/datasets/DATA-001.yaml")
    train_dataset = MonReaderDirectoryDataset(args.image_root, train_rows, transform)
    validation_dataset = MonReaderDirectoryDataset(args.image_root, validation_rows, transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_dataloader_worker if args.workers > 0 else None,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=make_torch_generator(seed),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=make_torch_generator(seed),
        **loader_kwargs,
    )
    config = ScratchCNNConfig(depth=2, start_filters=8, dropout=0.1)
    model = ScratchCNN(config)
    optimizer = build_optimizer(
        model.parameters(),
        name="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    device = torch.device("cpu")
    started = time.perf_counter()
    result = run_binary_training(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device=device,
        checkpoint_path=args.checkpoint,
        config={
            "purpose": "cpu_feasibility_only",
            "depth": config.depth,
            "start_filters": config.start_filters,
            "dropout": config.dropout,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": args.batch_size,
            "seed": seed,
            "augmentation": False,
            "threshold": 0.5,
        },
        max_epochs=1,
        patience=8,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "status": "ok",
        "device": "cpu",
        "purpose": "feasibility_only_not_scientific_result",
        "elapsed_seconds": elapsed,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "parameter_count": count_trainable_parameters(model),
        "best_epoch": result["best_epoch"],
        "best_validation_f1": result["best_validation_f1"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "history": result["history"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "elapsed_seconds", "train_count", "validation_count", "parameter_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
