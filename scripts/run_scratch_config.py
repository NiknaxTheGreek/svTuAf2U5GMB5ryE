from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MonReaderDirectoryDataset, join_manifest_with_membership
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.tuning import scratch_parameter_count
from src.preprocessing import build_base_transform
from src.training import build_optimizer, evaluate_binary_model, run_binary_training
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


REQUIRED_CONFIG_KEYS = {
    "trial_id",
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "dropout",
    "depth",
    "start_filters",
    "seed",
}


def load_frozen_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen scratch config must be a JSON object")
    missing = REQUIRED_CONFIG_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Frozen scratch config missing keys: {sorted(missing)}")
    if str(payload["optimizer"]) not in {"adam", "adamw", "sgd", "rmsprop"}:
        raise ValueError("Unsupported optimizer")
    if not 1e-5 <= float(payload["learning_rate"]) <= 1e-2:
        raise ValueError("Learning rate outside global scratch bounds")
    if not 8 <= int(payload["batch_size"]) <= 256:
        raise ValueError("Batch size outside global scratch bounds")
    wd = float(payload["weight_decay"])
    if wd != 0.0 and not 1e-6 <= wd <= 1e-2:
        raise ValueError("Weight decay outside global scratch bounds")
    ScratchCNNConfig(
        depth=int(payload["depth"]),
        start_filters=int(payload["start_filters"]),
        dropout=float(payload["dropout"]),
    )
    expected_parameters = scratch_parameter_count(
        int(payload["depth"]), int(payload["start_filters"])
    )
    if "parameter_count" in payload and int(payload["parameter_count"]) != expected_parameters:
        raise ValueError(
            f"Parameter-count mismatch in frozen config: {payload['parameter_count']} != {expected_parameters}"
        )
    payload["parameter_count"] = expected_parameters
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one frozen scratch-CNN configuration.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", required=True)
    return parser.parse_args()


def classify_failure(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, MemoryError) or "out of memory" in message or "cannot allocate memory" in message:
        return "out_of_memory"
    return type(exc).__name__


def main() -> int:
    args = parse_args()
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

        transform = build_base_transform("manifests/datasets/DATA-001.yaml")
        train_dataset = MonReaderDirectoryDataset(args.image_root, train_rows, transform)
        validation_dataset = MonReaderDirectoryDataset(args.image_root, validation_rows, transform)
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
        expected_parameters = config.get("parameter_count")
        if expected_parameters is not None and observed_parameters != int(expected_parameters):
            raise ValueError(
                f"Parameter-count mismatch: {observed_parameters} != {expected_parameters}"
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
            max_epochs=50,
            patience=8,
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = evaluate_binary_model(model, validation_loader, device=device)
        reloaded_f1 = float(validation["metrics"]["f1"])
        if abs(reloaded_f1 - float(training["best_validation_f1"])) > 1e-12:
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
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
