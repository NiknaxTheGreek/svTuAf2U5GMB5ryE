from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_scratch_config import load_frozen_config
from src.data import MonReaderMemmapDataset, join_manifest_with_membership
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.training import (
    build_optimizer,
    evaluate_binary_model,
    save_training_checkpoint,
    train_one_epoch,
)
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


def relative_change(current: float, previous: float) -> float:
    return abs(current - previous) / max(abs(previous), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continue one frozen scratch-CNN configuration until its learning curves plateau."
    )
    parser.add_argument("--cache-data", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    parser.add_argument("--stable-checks", type=int, default=2)
    args = parser.parse_args()

    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        raise ValueError("Require 1 <= min_epochs <= max_epochs")
    if args.window < 2 or args.stable_checks < 1:
        raise ValueError("window must be >=2 and stable-checks must be positive")
    if args.relative_tolerance <= 0:
        raise ValueError("relative-tolerance must be positive")

    config = load_frozen_config(args.config)
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
        raise ValueError("Membership must contain non-empty fit and validation populations")

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

    history: list[dict[str, object]] = []
    best_f1 = float("-inf")
    best_epoch = 0
    best_checkpoint_sha256: str | None = None
    stable_count = 0
    convergence_checks: list[dict[str, float | int | bool]] = []
    stop_reason = "max_epochs_cap"
    started = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device=device, amp_enabled=False
        )
        validation = evaluate_binary_model(model, validation_loader, device=device)
        validation_metrics = validation["metrics"]

        improved = float(validation_metrics["f1"]) > best_f1
        if improved:
            best_f1 = float(validation_metrics["f1"])
            best_epoch = epoch
            best_checkpoint_sha256 = save_training_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_f1=best_f1,
                config=config,
            )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "improved": improved,
            }
        )

        converged = False
        if epoch >= max(args.min_epochs, 2 * args.window):
            previous = history[-2 * args.window : -args.window]
            current = history[-args.window :]
            prev_train = float(np.mean([float(x["train"]["loss"]) for x in previous]))
            curr_train = float(np.mean([float(x["train"]["loss"]) for x in current]))
            prev_val = float(np.mean([float(x["validation"]["loss"]) for x in previous]))
            curr_val = float(np.mean([float(x["validation"]["loss"]) for x in current]))
            train_change = relative_change(curr_train, prev_train)
            validation_change = relative_change(curr_val, prev_val)
            stable = (
                train_change <= args.relative_tolerance
                and validation_change <= args.relative_tolerance
            )
            stable_count = stable_count + 1 if stable else 0
            converged = stable_count >= args.stable_checks
            convergence_checks.append(
                {
                    "epoch": epoch,
                    "train_loss_relative_change": train_change,
                    "validation_loss_relative_change": validation_change,
                    "stable": stable,
                    "stable_count": stable_count,
                    "converged": converged,
                }
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} "
            f"validation_f1={validation_metrics['f1']:.6f} "
            f"stable_count={stable_count}",
            flush=True,
        )
        if converged:
            stop_reason = "learning_curves_plateau"
            break

    final_validation = evaluate_binary_model(model, validation_loader, device=device)["metrics"]
    if best_checkpoint_sha256 is None or best_epoch < 1:
        raise RuntimeError("No valid best checkpoint was created")

    best_checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_validation = evaluate_binary_model(model, validation_loader, device=device)["metrics"]

    result = {
        "experiment": "perfect_config_convergence",
        "source_candidate_id": config.get("source_candidate_id"),
        "trial_id": config["trial_id"],
        "config": config,
        "membership": args.membership.as_posix(),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "parameter_count": observed_parameters,
        "seed": seed,
        "min_epochs": args.min_epochs,
        "max_epochs_cap": args.max_epochs,
        "convergence_window": args.window,
        "relative_tolerance": args.relative_tolerance,
        "required_stable_checks": args.stable_checks,
        "converged": stop_reason == "learning_curves_plateau",
        "stop_reason": stop_reason,
        "stopped_epoch": len(history),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "best_validation_metrics": best_validation,
        "final_epoch_validation_metrics": final_validation,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "convergence_checks": convergence_checks,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "test_set_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.checkpoint.exists():
        args.checkpoint.unlink()

    print(
        json.dumps(
            {
                "trial_id": result["trial_id"],
                "converged": result["converged"],
                "stop_reason": result["stop_reason"],
                "stopped_epoch": result["stopped_epoch"],
                "best_epoch": result["best_epoch"],
                "best_validation_f1": result["best_validation_f1"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
