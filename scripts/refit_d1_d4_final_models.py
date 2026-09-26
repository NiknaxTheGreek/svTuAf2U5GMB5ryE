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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 2026
IMAGE_SIZE = 160
BATCH_SIZE = 64
HEAD_EPOCHS = 2
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


class TrainDataset(Dataset):
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
        return x, y


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
    return total / len(loader.dataset)


def build_model() -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    for p in model.features.parameters():
        p.requires_grad = False
    return model


def train_dataset_model(
    dataset_id: str,
    train_df: pd.DataFrame,
    selected_epoch: int,
    images_root: Path,
    out_dir: Path,
) -> dict:
    set_determinism(SEED)
    target_map = {"notflip": 0, "flip": 1}
    train_df = train_df.copy()
    train_df["target"] = train_df["label"].map(target_map)
    if train_df["target"].isna().any():
        raise SystemExit(f"{dataset_id}: unknown label")

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

    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TrainDataset(train_df, images_root, train_transform)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        generator=generator,
        persistent_workers=True,
    )

    device = torch.device("cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )

    history = []
    for epoch in range(1, selected_epoch + 1):
        if epoch == HEAD_EPOCHS + 1:
            for block in model.features[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=2e-4,
                weight_decay=1e-4,
            )
        loss = train_one_epoch(model, loader, optimizer, device)
        history.append({
            "dataset_id": dataset_id,
            "epoch": epoch,
            "phase": "head" if epoch <= HEAD_EPOCHS else "finetune",
            "train_loss": loss,
        })
        print(json.dumps(history[-1]))

    checkpoint_path = out_dir / f"{dataset_id.lower()}_mobilenetv3_small.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "dataset_id": dataset_id,
        "selected_epoch": selected_epoch,
        "architecture": "mobilenet_v3_small",
        "weights": "IMAGENET1K_V1",
        "image_size": IMAGE_SIZE,
        "threshold": 0.5,
        "seed": SEED,
        "training_frames": len(train_df),
        "training_flip_frames": int((train_df["label"] == "flip").sum()),
        "training_notflip_frames": int((train_df["label"] == "notflip").sum()),
    }, checkpoint_path)

    history_path = out_dir / f"{dataset_id.lower()}_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    return {
        "dataset_id": dataset_id,
        "training_frames": len(train_df),
        "training_flip_frames": int((train_df["label"] == "flip").sum()),
        "training_notflip_frames": int((train_df["label"] == "notflip").sum()),
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history": history_path.name,
        "training_history_sha256": sha256_file(history_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--epoch-selection", required=True, type=Path)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--d2-manifest", required=True, type=Path)
    ap.add_argument("--d3-manifest", required=True, type=Path)
    ap.add_argument("--d4-manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("Authoritative archive SHA mismatch")

    selection = json.loads(args.epoch_selection.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if selection["status"] != "COMPLETE":
        raise SystemExit("Epoch selection is not complete")
    if selection["protocol_id"] != protocol["protocol_id"]:
        raise SystemExit("Epoch selection/protocol mismatch")
    selected_epoch = int(selection["selected_epoch"])
    if selected_epoch not in protocol["selection_rule"]["candidate_epochs"]:
        raise SystemExit("Selected epoch outside frozen candidate set")
    if protocol["fixed_model_recipe"]["decision_threshold"] != 0.5:
        raise SystemExit("Frozen threshold changed")

    base = pd.read_csv(args.d2_manifest, dtype={"video_id": str})
    d3 = pd.read_csv(args.d3_manifest, dtype={"video_id": str})
    d4 = pd.read_csv(args.d4_manifest, dtype={"video_id": str})

    if len(base) != 2989 or len(d3) != 1912 or len(d4) != 1912:
        raise SystemExit("Frozen dataset populations changed")

    datasets = {
        "D1": base.loc[base["supplied_split"] == "training"].copy(),
        "D2": base.loc[base["d2_split"] == "train"].copy(),
        "D3": d3.loc[d3["d3_split"] == "training"].copy(),
        "D4": d4.loc[d4["d4_split"] == "train"].copy(),
    }
    expected_counts = {"D1": 2392, "D2": 2392, "D3": 1521, "D4": 1504}
    observed = {k: len(v) for k, v in datasets.items()}
    if observed != expected_counts:
        raise SystemExit(f"Final training counts changed: {observed}")

    # Metadata-only test population checks. No test Dataset or Image.open is created here.
    tests = {
        "D1": base.loc[base["supplied_split"] == "testing"],
        "D2": base.loc[base["d2_split"] == "test"],
        "D3": d3.loc[d3["d3_split"] == "testing"],
        "D4": d4.loc[d4["d4_split"] == "test"],
    }
    expected_test = {"D1": 597, "D2": 597, "D3": 391, "D4": 408}
    if {k: len(v) for k, v in tests.items()} != expected_test:
        raise SystemExit("Frozen test counts changed")

    images_root = resolve_images_root(args.dataset_root)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    models_meta = []
    for dataset_id in ("D1", "D2", "D3", "D4"):
        models_meta.append(
            train_dataset_model(
                dataset_id, datasets[dataset_id], selected_epoch, images_root, out
            )
        )

    manifest = {
        "status": "FROZEN",
        "protocol_id": protocol["protocol_id"],
        "selected_epoch": selected_epoch,
        "architecture": "MobileNetV3-Small",
        "pretrained_weights": "ImageNet IMAGENET1K_V1",
        "seed": SEED,
        "threshold": 0.5,
        "models": models_meta,
        "information_boundary": {
            "D1_test_pixels_opened": False,
            "D2_test_pixels_opened": False,
            "D3_test_pixels_opened": False,
            "D4_test_pixels_opened": False,
            "test_metrics_computed": False,
        },
        "next_action": (
            "Evaluate all four frozen checkpoint hashes in one locked workflow. "
            "No model, threshold, preprocessing, or dataset changes are permitted "
            "between this freeze and evaluation."
        ),
        "input_sha256": {
            "epoch_selection": sha256_file(args.epoch_selection),
            "protocol": sha256_file(args.protocol),
            "d2_manifest": sha256_file(args.d2_manifest),
            "d3_manifest": sha256_file(args.d3_manifest),
            "d4_manifest": sha256_file(args.d4_manifest),
            "archive": sha256_file(args.archive),
        },
    }
    manifest_path = out / "final_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "status": "PASS",
        "selected_epoch": selected_epoch,
        "model_checkpoint_hashes": {
            m["dataset_id"]: m["checkpoint_sha256"] for m in models_meta
        },
        "test_pixels_opened": False,
        "test_metrics_computed": False,
        "final_model_manifest_sha256": sha256_file(manifest_path),
    }
    receipt_path = out / "final_refit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
