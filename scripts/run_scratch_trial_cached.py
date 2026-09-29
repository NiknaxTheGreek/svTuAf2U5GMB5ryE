from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MonReaderMemmapDataset, join_manifest_with_membership
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.training import build_optimizer, evaluate_binary_model, run_binary_training
from src.tuning import validate_stage1_payload
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


def load_trial(path: Path, trial_id: str) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_stage1_payload(payload)
    matches = [trial for trial in payload["trials"] if trial["trial_id"] == trial_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen Stage-1 config for {trial_id}")
    fixed = payload.get("fixed")
    if not isinstance(fixed, dict):
        raise ValueError("Stage-1 fixed training settings are missing")
    return dict(matches[0]), dict(fixed)


def classify_failure(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, MemoryError) or "out of memory" in message or "cannot allocate memory" in message:
        return "out_of_memory"
    return type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one frozen Stage-1 trial from the exact preprocessing cache.")
    parser.add_argument("--cache-data", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--trial-config", required=True, type=Path)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("manifests/splits/SPLIT-007_original_tuning.csv"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    trial, fixed = load_trial(args.trial_config, args.trial_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, object] = {
        "trial_id": args.trial_id,
        "stage": "random",
        "status": "failed",
        "config": trial,
        "membership": args.membership.as_posix(),
        "device": "cpu",
        "preprocessing_cache": "exact_uint8_memmap",
        "fixed_training": fixed,
    }
    try:
        seed = int(trial["seed"])
        seed_everything(seed)
        rows = join_manifest_with_membership(
            "manifests/datasets/DATA-001.csv",
            args.membership,
            allowed_partitions={"fit", "validation"},
        )
        train_rows = [row for row in rows if row["partition"] == "fit"]
        validation_rows = [row for row in rows if row["partition"] == "validation"]
        if len(train_rows) != 2152 or len(validation_rows) != 240:
            raise ValueError("Stage-1 development population changed unexpectedly")

        train_dataset = MonReaderMemmapDataset(args.cache_data, args.cache_index, train_rows)
        validation_dataset = MonReaderMemmapDataset(args.cache_data, args.cache_index, validation_rows)
        loader_kwargs = {
            "batch_size": int(trial["batch_size"]),
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
                depth=int(trial["depth"]),
                start_filters=int(trial["start_filters"]),
                dropout=float(trial["dropout"]),
            )
        )
        observed_parameters = count_trainable_parameters(model)
        if observed_parameters != int(trial["parameter_count"]):
            raise ValueError(
                f"Parameter-count mismatch: {observed_parameters} != {trial['parameter_count']}"
            )
        optimizer = build_optimizer(
            model.parameters(),
            name=str(trial["optimizer"]),
            learning_rate=float(trial["learning_rate"]),
            weight_decay=float(trial["weight_decay"]),
        )
        device = torch.device("cpu")
        training = run_binary_training(
            model,
            train_loader,
            validation_loader,
            optimizer,
            device=device,
            checkpoint_path=args.checkpoint,
            config=trial,
            max_epochs=int(fixed["max_epochs"]),
            patience=int(fixed["early_stopping_patience"]),
        )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = evaluate_binary_model(model, validation_loader, device=device)
        if abs(float(validation["metrics"]["f1"]) - float(training["best_validation_f1"])) > 1e-12:
            raise RuntimeError("Reloaded checkpoint does not reproduce selected validation F1")
        result.update(
            {
                "status": "success",
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
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "best_validation_f1": result.get("best_validation_f1"),
                "failure_type": result.get("failure_type"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
