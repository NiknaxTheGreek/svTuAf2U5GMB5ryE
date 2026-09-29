from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.run_scratch_config import classify_failure, load_frozen_config
from src.data import MonReaderMemmapDataset, join_manifest_with_membership
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.training import build_optimizer, evaluate_binary_model, run_binary_training
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one frozen scratch-CNN config from an exact uint8 cache.")
    parser.add_argument("--cache-data", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    config = load_frozen_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, object] = {
        "trial_id": str(config["trial_id"]),
        "stage": args.stage,
        "status": "failed",
        "config": config,
        "membership": args.membership.as_posix(),
        "device": "cpu",
        "preprocessing_cache": "exact_uint8_memmap",
    }
    if "source_candidate_id" in config:
        result["source_candidate_id"] = str(config["source_candidate_id"])
    try:
        seed = int(config["seed"])
        seed_everything(seed)
        rows = join_manifest_with_membership(
            "manifests/datasets/DATA-001.csv",
            args.membership,
            allowed_partitions={"fit", "validation"},
        )
        train_rows = [row for row in rows if row["partition"] == "fit"]
        validation_rows = [row for row in rows if row["partition"] == "validation"]
        if not train_rows or not validation_rows:
            raise ValueError("Scratch membership must contain non-empty fit and validation populations")

        train_dataset = MonReaderMemmapDataset(args.cache_data, args.cache_index, train_rows)
        validation_dataset = MonReaderMemmapDataset(args.cache_data, args.cache_index, validation_rows)
        loader_kwargs = {
            "batch_size": int(config["batch_size"]),
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

        model = ScratchCNN(
            ScratchCNNConfig(
                depth=int(config["depth"]),
                start_filters=int(config["start_filters"]),
                dropout=float(config["dropout"]),
            )
        )
        observed_parameters = count_trainable_parameters(model)
        if observed_parameters != int(config["parameter_count"]):
            raise ValueError(
                f"Parameter-count mismatch: {observed_parameters} != {config['parameter_count']}"
            )
        optimizer = build_optimizer(
            model.parameters(),
            name=str(config["optimizer"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        device = torch.device("cpu")
        training = run_binary_training(
            model,
            train_loader,
            validation_loader,
            optimizer,
            device=device,
            checkpoint_path=args.checkpoint,
            config=config,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = evaluate_binary_model(model, validation_loader, device=device)
        if abs(float(validation["metrics"]["f1"]) - float(training["best_validation_f1"])) > 1e-12:
            raise RuntimeError("Reloaded checkpoint does not reproduce selected validation F1")
        result.update(
            {
                "status": "success",
                "seed": seed,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "parameter_count": observed_parameters,
                "best_epoch": training["best_epoch"],
                "best_validation_f1": training["best_validation_f1"],
                "best_validation_metrics": validation["metrics"],
                "checkpoint_sha256": training["checkpoint_sha256"],
                "history": training["history"],
            }
        )
    except Exception as exc:
        result.update(
            {
                "failure_type": classify_failure(exc),
                "failure_message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
            }
        )
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.checkpoint.exists():
            args.checkpoint.unlink()

    print(
        json.dumps(
            {
                "trial_id": result["trial_id"],
                "stage": result["stage"],
                "status": result["status"],
                "best_validation_f1": result.get("best_validation_f1"),
                "failure_type": result.get("failure_type"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
