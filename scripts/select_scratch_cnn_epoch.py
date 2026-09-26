#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 2026
IMAGE_SIZE = 160
BATCH_SIZE = 64
MAX_EPOCHS = 30
THRESHOLD = 0.5
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


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: Path, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(self.images_root / row["path"]) as im:
            x = self.transform(im.convert("RGB"))
        y = torch.tensor(float(row["target"]), dtype=torch.float32)
        return x, y, idx


class ScratchCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    probs, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy().astype(int))
    return np.concatenate(targets), np.concatenate(probs), total_loss / len(loader.dataset)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
    return total_loss / len(loader.dataset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--comparison-protocol", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    set_determinism(SEED)
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("Authoritative archive SHA mismatch")

    protocol = json.loads(args.comparison_protocol.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "CNN_FAMILY_COMPARISON_V1":
        raise SystemExit("Unexpected comparison protocol")
    scratch = protocol["families"]["scratch_cnn"]
    if scratch["status"] != "ARCHITECTURE_FROZEN_EPOCH_PENDING":
        raise SystemExit("Scratch architecture protocol status changed")
    if protocol["shared_rules"]["decision_threshold"] != THRESHOLD:
        raise SystemExit("Threshold changed from frozen protocol")

    df = pd.read_csv(args.manifest, dtype={"video_id": str})
    required = {
        "sequence_id", "label", "video_id", "frame_num", "path",
        "environment_group_id", "d2_split", "model_selection_role",
    }
    if missing := required - set(df.columns):
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    if len(df) != 2989 or df["path"].duplicated().any():
        raise SystemExit("Frozen population changed")

    train_df = df.loc[df["model_selection_role"] == "dev_train"].copy()
    val_df = df.loc[df["model_selection_role"] == "dev_validation"].copy()
    protected_df = df.loc[df["model_selection_role"] == "protected_test"].copy()
    if (len(train_df), len(val_df), len(protected_df)) != (1914, 478, 597):
        raise SystemExit("Frozen development counts changed")

    if set(train_df["environment_group_id"]) & set(val_df["environment_group_id"]):
        raise SystemExit("Development train/validation environment overlap")
    if set(train_df["sequence_id"]) & set(val_df["sequence_id"]):
        raise SystemExit("Development train/validation sequence overlap")
    if set(protected_df["environment_group_id"]) & (
        set(train_df["environment_group_id"]) | set(val_df["environment_group_id"])
    ):
        raise SystemExit("Protected test environment leaked into development")

    target_map = {"notflip": 0, "flip": 1}
    train_df["target"] = train_df["label"].map(target_map)
    val_df["target"] = val_df["label"].map(target_map)

    images_root = resolve_images_root(args.dataset_root)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.90, 1.00), ratio=(0.95, 1.05), antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        FrameDataset(train_df, images_root, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        generator=generator,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        FrameDataset(val_df, images_root, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    device = torch.device("cpu")
    model = ScratchCNN().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    history = []
    best_f1 = -1.0
    best_epoch = None
    best_metrics = None
    best_probs = None
    best_targets = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        y_val, p_val, val_loss = evaluate(model, val_loader, device)
        metrics = metric_block(y_val, p_val)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            **{f"validation_{k}": v for k, v in metrics.items() if k != "support"},
        }
        history.append(row)
        print(json.dumps(row))

        if metrics["f1"] > best_f1 + 1e-12:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_metrics = metrics
            best_probs = p_val.copy()
            best_targets = y_val.copy()

    if best_epoch is None:
        raise SystemExit("No scratch epoch selected")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    history_path = out / "scratch_epoch_selection_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    pred = val_df[[
        "path", "sequence_id", "environment_group_id", "label", "frame_num"
    ]].copy().reset_index(drop=True)
    pred["target"] = best_targets
    pred["prob_flip"] = best_probs
    pred["prediction"] = (best_probs >= THRESHOLD).astype(int)
    pred["correct"] = pred["target"] == pred["prediction"]
    pred_path = out / "scratch_epoch_selection_validation_predictions.csv"
    pred.to_csv(pred_path, index=False)

    result = {
        "status": "COMPLETE",
        "protocol_id": protocol["protocol_id"],
        "model_family": "scratch_cnn",
        "architecture": scratch["architecture"],
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "selected_epoch": int(best_epoch),
        "selected_by": "highest D2 development validation F1; earliest epoch wins ties",
        "validation_metrics": best_metrics,
        "candidate_epoch_count": MAX_EPOCHS,
        "threshold": THRESHOLD,
        "threshold_tuned": False,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "selection_population": {
            "dev_train_frames": len(train_df),
            "dev_validation_frames": len(val_df),
            "protected_test_frames": len(protected_df),
            "protected_test_pixels_opened": False,
            "protected_test_metrics_computed": False,
        },
        "final_refit_instruction": (
            "Use the selected scratch epoch unchanged for D1-D4. Reinitialize the "
            "scratch CNN under seed 2026 for every dataset and train on each full "
            "non-test training portion. Do not reuse the selection model."
        ),
        "artifact_sha256": {
            history_path.name: sha256_file(history_path),
            pred_path.name: sha256_file(pred_path),
        },
    }
    result_path = out / "scratch_epoch_selection.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "status": "PASS",
        "inputs": {
            str(args.manifest): sha256_file(args.manifest),
            str(args.comparison_protocol): sha256_file(args.comparison_protocol),
            str(args.archive): sha256_file(args.archive),
        },
        "outputs": {
            result_path.name: sha256_file(result_path),
            history_path.name: sha256_file(history_path),
            pred_path.name: sha256_file(pred_path),
        },
        "protected_test_pixels_opened": False,
        "protected_test_metrics_computed": False,
    }
    receipt_path = out / "scratch_epoch_selection_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
