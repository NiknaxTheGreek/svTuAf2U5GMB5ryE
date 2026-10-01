from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from src.resnet18_v2 import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_COUNTS,
    build_model,
    load_fixed_config,
    set_head_only,
    trainable_parameter_count,
    unfreeze_all,
)
from src.scratch_v2 import EXPECTED_SPLIT_SHA256, sha256_file


def main() -> int:
    cfg_path = Path("configs/RESNET18_FIXED_COMPARATOR_V2.json")
    cfg = load_fixed_config(cfg_path)
    if sha256_file(cfg_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Frozen comparator config hash changed")

    for regime in ("O","S","T","ST"):
        path = Path(f"manifests/splits/{regime}.csv")
        if sha256_file(path) != EXPECTED_SPLIT_SHA256[regime]:
            raise RuntimeError(f"{regime} split SHA changed")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter(r["role"] for r in rows)
        if counts["train"] != EXPECTED_COUNTS[(regime,"train")] or counts["test"] != EXPECTED_COUNTS[(regime,"test")]:
            raise RuntimeError(f"{regime} train/test count mismatch")

    torch.manual_seed(int(cfg["seed"]))
    model = build_model(cfg, pretrained=False)
    set_head_only(model)
    if trainable_parameter_count(model) != 513:
        raise RuntimeError("Head-only phase must expose exactly the one-logit replacement head")
    x = torch.zeros(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0])
    model.eval()
    model.fc.train()
    logits = model(x).squeeze(1)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    unfreeze_all(model)
    if trainable_parameter_count(model) <= 11_000_000:
        raise RuntimeError("Full-network unfreeze failed")

    print(json.dumps({
        "status": "PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "regimes": ["O","S","T","ST"],
        "head_only_trainable_parameters": 513,
        "synthetic_forward_backward": True,
        "validation_used": False,
        "test_used": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
