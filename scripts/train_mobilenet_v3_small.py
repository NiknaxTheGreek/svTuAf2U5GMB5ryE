#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 2026
IMAGE_SIZE = 160
BATCH_SIZE = 64
HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 6
TOTAL_EPOCHS = HEAD_EPOCHS + FINETUNE_EPOCHS
POSITIVE_LABEL = 1
NAME_RE = re.compile(r"^(?P<video_id>.+)_(?P<frame_num>\d+)$")
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_images_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "training").is_dir() and (root / "testing").is_dir():
        return root
    if (root / "images" / "training").is_dir() and (root / "images" / "testing").is_dir():
        return root / "images"
    raise FileNotFoundError(f"Could not resolve image root from {root}")


def build_manifest(images_root: Path, split_csv: Path) -> pd.DataFrame:
    frozen = pd.read_csv(split_csv, dtype={"video_id": str, "source_group_id": str})
    expected_cols = {"source_group_id", "label", "target", "video_id", "frame_count", "partition"}
    if missing := expected_cols - set(frozen.columns):
        raise ValueError(f"Frozen split missing columns: {sorted(missing)}")
    if len(frozen) != 117:
        raise AssertionError(f"Expected 117 frozen source groups, got {len(frozen)}")
    if frozen["source_group_id"].duplicated().any():
        raise AssertionError("Frozen split has duplicated source_group_id")

    partition_by_group = dict(zip(frozen["source_group_id"], frozen["partition"]))
    target_by_label = {"notflip": 0, "flip": 1}

    rows = []
    for supplied_split in ("training", "testing"):
        for label in ("flip", "notflip"):
            folder = images_root / supplied_split / label
            for path in sorted(folder.glob("*.jpg")):
                m = NAME_RE.match(path.stem)
                if not m:
                    raise ValueError(f"Unexpected filename: {path.name}")
                video_id = f"{int(m.group('video_id')):04d}"
                frame_num = int(m.group("frame_num"))
                source_group_id = f"{label}__{video_id}"
                if source_group_id not in partition_by_group:
                    raise AssertionError(f"Group missing from frozen split: {source_group_id}")
                rows.append(
                    {
                        "path": str(path),
                        "relative_path": path.relative_to(images_root).as_posix(),
                        "label": label,
                        "target": target_by_label[label],
                        "video_id": video_id,
                        "frame_num": frame_num,
                        "source_group_id": source_group_id,
                        "partition": partition_by_group[source_group_id],
                    }
                )

    manifest = pd.DataFrame(rows)
    if len(manifest) != 2989:
        raise AssertionError(f"Expected 2,989 frames, got {len(manifest)}")

    expected_partition_counts = {"train": 1795, "validation": 597, "test": 597}
    observed = manifest.groupby("partition").size().to_dict()
    if observed != expected_partition_counts:
        raise AssertionError(f"Frozen frame counts changed: {observed}")

    if manifest.duplicated(["label", "video_id", "frame_num"]).any():
        raise AssertionError("Duplicate canonical source-frame key found")

    # Verify that no source group is assigned to multiple partitions.
    group_partition_counts = manifest.groupby("source_group_id")["partition"].nunique()
    if int(group_partition_counts.max()) != 1:
        raise AssertionError("Source-group leakage detected in manifest")

    return manifest.sort_values(
        ["partition", "target", "source_group_id", "frame_num"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


class FrameDataset(Dataset):
    def __init__(self, frame_df: pd.DataFrame, transform) -> None:
        self.rows = frame_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        y = torch.tensor(float(row["target"]), dtype=torch.float32)
        return x, y, index


def metric_block(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(len(y_true)),
    }


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, targets = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy().astype(int))
    return (
        np.concatenate(targets),
        np.concatenate(probs),
        total_loss / len(loader.dataset),
    )


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--archive", required=False, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    set_determinism(SEED)

    if args.archive:
        actual_sha = sha256_file(args.archive)
        if actual_sha != EXPECTED_ARCHIVE_SHA256:
            raise AssertionError(
                f"Authoritative archive SHA mismatch: {actual_sha} != {EXPECTED_ARCHIVE_SHA256}"
            )

    images_root = resolve_images_root(args.dataset_root)
    manifest = build_manifest(images_root, args.split)

    # Freeze test information boundary before any Image.open calls.
    train_df = manifest.loc[manifest["partition"] == "train"].copy()
    val_df = manifest.loc[manifest["partition"] == "validation"].copy()
    test_df = manifest.loc[manifest["partition"] == "test"].copy()
    if (len(train_df), len(val_df), len(test_df)) != (1795, 597, 597):
        raise AssertionError("Frozen partition sizes changed")

    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    baseline_f1 = float(
        baseline.get("locked_validation_baseline_f1", baseline["validation"]["f1"])
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.90, 1.00),
                ratio=(0.95, 1.05),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_ds = FrameDataset(train_df, train_transform)
    val_ds = FrameDataset(val_df, eval_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        generator=generator,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    device = torch.device("cpu")

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    # Phase 1: train only the new classifier head.
    for p in model.features.parameters():
        p.requires_grad = False
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_mobilenetv3_small.pt"

    history = []
    best_f1 = -1.0
    best_epoch = None
    best_metrics = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        if epoch == HEAD_EPOCHS + 1:
            # Phase 2: unfreeze the last four feature blocks for limited fine-tuning.
            for block in model.features[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=2e-4,
                weight_decay=1e-4,
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        y_val, p_val, val_loss = evaluate(model, val_loader, device)
        val_metrics = metric_block(y_val, p_val, threshold=0.5)

        row = {
            "epoch": epoch,
            "phase": "head" if epoch <= HEAD_EPOCHS else "finetune",
            "train_loss": train_loss,
            "validation_loss": val_loss,
            **{f"validation_{k}": v for k, v in val_metrics.items() if k != "support"},
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["f1"] > best_f1 + 1e-12:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "architecture": "mobilenet_v3_small",
                    "weights": "IMAGENET1K_V1",
                    "image_size": IMAGE_SIZE,
                    "threshold": 0.5,
                    "seed": SEED,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_val, p_val, val_loss = evaluate(model, val_loader, device)
    final_metrics = metric_block(y_val, p_val, threshold=0.5)
    if abs(final_metrics["f1"] - best_f1) > 1e-12:
        raise AssertionError("Reloaded best checkpoint does not reproduce best validation F1")

    val_predictions = val_df[
        ["relative_path", "source_group_id", "label", "target", "frame_num"]
    ].copy()
    val_predictions["prob_flip"] = p_val
    val_predictions["prediction"] = (p_val >= 0.5).astype(int)
    val_predictions["correct"] = (
        val_predictions["target"] == val_predictions["prediction"]
    )

    errors = val_predictions.loc[~val_predictions["correct"]].copy()
    error_by_group = (
        errors.groupby(["source_group_id", "label"])
        .agg(errors=("correct", "size"))
        .reset_index()
        .sort_values(["errors", "source_group_id"], ascending=[False, True])
    )

    history_path = out / "training_history.csv"
    pred_path = out / "validation_predictions.csv"
    errors_path = out / "validation_errors_by_group.csv"
    manifest_path = out / "development_manifest.csv"
    metrics_path = out / "model_metrics.json"

    pd.DataFrame(history).to_csv(history_path, index=False)
    val_predictions.to_csv(pred_path, index=False)
    error_by_group.to_csv(errors_path, index=False)
    manifest.loc[manifest["partition"].isin(["train", "validation"])].drop(
        columns=["path"]
    ).to_csv(manifest_path, index=False)

    result = {
        "status": "COMPLETE",
        "architecture": "MobileNetV3-Small",
        "pretrained_weights": "ImageNet IMAGENET1K_V1",
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "training_schedule": {
            "head_epochs": HEAD_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "total_epochs": TOTAL_EPOCHS,
            "head_learning_rate": 0.001,
            "finetune_learning_rate": 0.0002,
            "unfrozen_feature_blocks": 4,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,
        },
        "selection": {
            "selected_by": "highest validation F1 across epochs",
            "decision_threshold": 0.5,
            "threshold_tuned": False,
            "best_epoch": best_epoch,
        },
        "validation": final_metrics,
        "baseline_validation_f1": baseline_f1,
        "f1_delta_vs_baseline": float(final_metrics["f1"] - baseline_f1),
        "beats_baseline": bool(final_metrics["f1"] > baseline_f1),
        "information_boundary": {
            "train_frames_opened": len(train_df),
            "validation_frames_opened": len(val_df),
            "test_frames_in_manifest": len(test_df),
            "test_pixels_opened": False,
            "test_metrics_computed": False,
        },
        "artifact_sha256": {
            "best_mobilenetv3_small.pt": sha256_file(best_path),
            "training_history.csv": sha256_file(history_path),
            "validation_predictions.csv": sha256_file(pred_path),
            "validation_errors_by_group.csv": sha256_file(errors_path),
            "development_manifest.csv": sha256_file(manifest_path),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
