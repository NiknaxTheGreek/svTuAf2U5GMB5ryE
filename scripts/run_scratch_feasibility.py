from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MonReaderZipDataset, join_manifest_with_membership, verify_archive_identity
from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters
from src.preprocessing import build_base_transform
from src.training import build_optimizer, evaluate_binary_model, run_binary_training
from src.utils import make_torch_generator, seed_dataloader_worker, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full-population scratch-CNN feasibility smoke.")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/datasets/DATA-001.csv"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("manifests/splits/SPLIT-007_original_tuning.csv"),
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path("manifests/datasets/DATA-001.yaml"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_archive_identity(args.archive)
    seed = 42
    seed_everything(seed)

    fit_rows = join_manifest_with_membership(
        args.manifest, args.membership, allowed_partitions={"fit"}
    )
    validation_rows = join_manifest_with_membership(
        args.manifest, args.membership, allowed_partitions={"validation"}
    )
    if len(fit_rows) != 2152 or len(validation_rows) != 240:
        raise ValueError(
            f"Unexpected tuning population sizes: fit={len(fit_rows)}, validation={len(validation_rows)}"
        )

    transform = build_base_transform(args.registration)
    fit_dataset = MonReaderZipDataset(args.archive, fit_rows, transform)
    validation_dataset = MonReaderZipDataset(args.archive, validation_rows, transform)

    batch_size = 64
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=make_torch_generator(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model_config = ScratchCNNConfig(depth=1, start_filters=8, dropout=0.1)
    model = ScratchCNN(model_config)
    device = torch.device("cpu")
    model.to(device)
    optimizer = build_optimizer(
        model.parameters(),
        name="adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.9,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "feasibility_best.pt"
    run_config = {
        "purpose": "engineering_feasibility_only",
        "scientific_result": False,
        "seed": seed,
        "device": "cpu",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "weight_decay": 0.0,
        "dropout": model_config.dropout,
        "depth": model_config.depth,
        "start_filters": model_config.start_filters,
        "max_epochs": 1,
        "patience": 8,
        "augmentation": False,
        "scheduler": None,
        "gradient_clipping": None,
    }
    training = run_binary_training(
        model,
        fit_loader,
        validation_loader,
        optimizer,
        device=device,
        checkpoint_path=checkpoint_path,
        config=run_config,
        max_epochs=1,
        patience=8,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_binary_model(model, validation_loader, device=device)
    metrics = validation["metrics"]

    if metrics["n"] != 240:
        raise RuntimeError("Feasibility validation did not cover all 240 samples")
    if training["best_epoch"] != 1:
        raise RuntimeError("One-epoch feasibility run did not save epoch 1")
    if not 0.0 <= float(metrics["f1"]) <= 1.0:
        raise RuntimeError("Validation F1 is outside [0,1]")

    result = {
        "status": "pass",
        "purpose": "engineering_feasibility_only",
        "scientific_result": False,
        "fit_count": len(fit_rows),
        "validation_count": len(validation_rows),
        "parameter_count": count_trainable_parameters(model),
        "best_epoch": training["best_epoch"],
        "best_validation_f1": training["best_validation_f1"],
        "validation_metrics": metrics,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "config": run_config,
    }
    (output_root / "feasibility_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    prediction_path = output_root / "feasibility_validation_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "sample_id",
            "video_id",
            "frame_number",
            "true_class",
            "probability_flip",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(validation["sample_ids"])):
            writer.writerow(
                {
                    "sample_id": validation["sample_ids"][index],
                    "video_id": validation["video_ids"][index],
                    "frame_number": validation["frame_numbers"][index],
                    "true_class": validation["labels"][index],
                    "probability_flip": validation["probabilities"][index],
                }
            )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
