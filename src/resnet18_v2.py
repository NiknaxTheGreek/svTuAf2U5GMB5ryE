from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.scratch_v2 import EXPECTED_SPLIT_SHA256, sha256_file

EXPECTED_CONFIG_SHA256 = "82af40ff6071949ee91d8934341b42bbdd7391738e0cde23d8ad9b9084338cc8"
EXPECTED_COUNTS = {
    ("O", "train"): 2392, ("O", "test"): 597,
    ("S", "train"): 2219, ("S", "test"): 770,
    ("T", "train"): 2365, ("T", "test"): 624,
    ("ST", "train"): 1756, ("ST", "test"): 161,
}
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_fixed_config(path: str | Path) -> dict:
    path = Path(path)
    digest = sha256_file(path)
    if digest != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"ResNet18 comparator config SHA mismatch: {digest}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "optimizer": "AdamW",
        "batch_size": 32,
        "head_learning_rate": 0.001,
        "finetune_learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "seed": 42,
        "threshold": 0.5,
        "validation": False,
        "early_stopping": False,
        "augmentation": False,
        "checkpoint": "final_epoch_20",
        "selection_bearing": False,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Frozen ResNet18 config field changed: {key}")
    if cfg["epochs"] != {"head_only": 5, "full_network": 15, "total": 20}:
        raise ValueError("Frozen ResNet18 epoch schedule changed")
    if cfg["selected_regimes"] != ["O", "S", "T", "ST"]:
        raise ValueError("Frozen ResNet18 regime list changed")
    return cfg


def build_model(cfg: dict, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(float(cfg["head"]["dropout"])),
        nn.Linear(in_features, 1),
    )
    return model


def set_head_only(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ResNetMemmapDataset(Dataset):
    def __init__(self, data_path: str | Path, index_path: str | Path):
        self.rows = []
        with Path(index_path).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        n = len(self.rows)
        self.data = np.lib.format.open_memmap(
            Path(data_path), mode="r", dtype=np.uint8, shape=(n, 3, 224, 224)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        x = torch.from_numpy(np.array(self.data[idx], copy=True)).float().div_(255.0)
        x = (x - MEAN) / STD
        label = 1.0 if self.rows[idx]["label"] == "flip" else 0.0
        return {
            "image": x,
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": self.rows[idx]["sample_id"],
        }


def verify_cache_receipt(receipt: dict, regime: str, role: str, data_path: Path, index_path: Path) -> None:
    if receipt["regime"] != regime or receipt["role"] != role:
        raise ValueError("ResNet18 cache regime/role mismatch")
    if receipt["rows"] != EXPECTED_COUNTS[(regime, role)]:
        raise ValueError("ResNet18 cache row count mismatch")
    if receipt["split_sha256"] != EXPECTED_SPLIT_SHA256[regime]:
        raise ValueError("ResNet18 cache split identity mismatch")
    if sha256_file(data_path) != receipt["cache_data_sha256"]:
        raise ValueError("ResNet18 cache data hash mismatch")
    if sha256_file(index_path) != receipt["cache_index_sha256"]:
        raise ValueError("ResNet18 cache index hash mismatch")
