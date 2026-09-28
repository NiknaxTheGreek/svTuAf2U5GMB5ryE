from __future__ import annotations

import logging
import os
import random
from typing import Final

import numpy as np
import torch


PUBLIC_NAMES: Final[dict[str, str]] = {
    "original": "Original",
    "random_stratified": "Random Stratified",
    "source_safe": "Source-Safe",
    "temporal_ordered": "Temporal-Ordered",
    "source_safe_temporal": "Source-Safe + Temporal",
    "flip": "flip",
    "notflip": "notflip",
}

METRIC_DECIMALS: Final[dict[str, int]] = {
    "f1": 3,
    "precision": 3,
    "recall": 3,
    "accuracy": 3,
    "balanced_accuracy": 3,
    "roc_auc": 3,
    "pr_auc": 3,
    "loss": 4,
}


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(choice: str = "auto") -> torch.device:
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device choice: {choice}")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
