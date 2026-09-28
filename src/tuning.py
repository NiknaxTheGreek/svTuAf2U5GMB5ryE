from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
STAGE1_SEED = 42
STAGE1_TRIAL_COUNT = 40
WEIGHT_DECAY_ZERO_PROBABILITY = 0.2


@dataclass(frozen=True)
class ScratchTrial:
    trial_id: str
    optimizer: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    dropout: float
    depth: int
    start_filters: int
    seed: int
    parameter_count: int


def scratch_parameter_count(depth: int, start_filters: int) -> int:
    if not 1 <= depth <= 7:
        raise ValueError("depth must be in [1, 7]")
    if not 8 <= start_filters <= 64:
        raise ValueError("start_filters must be in [8, 64]")
    total = 0
    in_channels = 3
    out_channels = start_filters
    for _ in range(depth):
        total += out_channels * in_channels * 3 * 3
        total += 2 * out_channels
        in_channels = out_channels
        out_channels *= 2
    total += in_channels + 1
    return total


def generate_stage1_random_trials(
    *,
    seed: int = STAGE1_SEED,
    trial_count: int = STAGE1_TRIAL_COUNT,
    weight_decay_zero_probability: float = WEIGHT_DECAY_ZERO_PROBABILITY,
) -> list[ScratchTrial]:
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    if not 0.0 <= weight_decay_zero_probability <= 1.0:
        raise ValueError("weight_decay_zero_probability must be in [0,1]")
    rng = np.random.default_rng(seed)
    trials: list[ScratchTrial] = []
    for index in range(1, trial_count + 1):
        optimizer = str(rng.choice(OPTIMIZERS))
        learning_rate = float(10.0 ** rng.uniform(-5.0, -2.0))
        batch_size = int(rng.integers(8, 257))
        weight_decay = (
            0.0
            if rng.random() < weight_decay_zero_probability
            else float(10.0 ** rng.uniform(-6.0, -2.0))
        )
        dropout = float(rng.uniform(0.0, 0.5))
        depth = int(rng.integers(1, 8))
        start_filters = int(rng.integers(8, 65))
        trials.append(
            ScratchTrial(
                trial_id=f"stage1-{index:03d}",
                optimizer=optimizer,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                dropout=dropout,
                depth=depth,
                start_filters=start_filters,
                seed=seed,
                parameter_count=scratch_parameter_count(depth, start_filters),
            )
        )
    return trials


def stage1_payload() -> dict[str, Any]:
    trials = generate_stage1_random_trials()
    return {
        "stage": "random",
        "seed": STAGE1_SEED,
        "trial_count": STAGE1_TRIAL_COUNT,
        "weight_decay_zero_probability": WEIGHT_DECAY_ZERO_PROBABILITY,
        "search_space": {
            "optimizer": list(OPTIMIZERS),
            "learning_rate": {
                "distribution": "log_uniform",
                "min": 1e-5,
                "max": 1e-2,
            },
            "batch_size": {
                "distribution": "integer_uniform",
                "min": 8,
                "max": 256,
            },
            "weight_decay": {
                "distribution": "mixture",
                "zero_probability": WEIGHT_DECAY_ZERO_PROBABILITY,
                "positive_distribution": "log_uniform",
                "positive_min": 1e-6,
                "positive_max": 1e-2,
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5,
            },
            "depth": {
                "distribution": "integer_uniform",
                "min": 1,
                "max": 7,
            },
            "start_filters": {
                "distribution": "integer_uniform",
                "min": 8,
                "max": 64,
            },
        },
        "fixed": {
            "seed": 42,
            "augmentation": False,
            "threshold": 0.5,
            "max_epochs": 50,
            "early_stopping_patience": 8,
            "scheduler": None,
            "gradient_clipping": None,
            "loss": "BCEWithLogitsLoss",
            "class_sampling": "natural",
            "canvas": {"width": 224, "height": 398},
            "scaling": "[0,1]",
        },
        "trials": [asdict(trial) for trial in trials],
    }


def validate_stage1_payload(payload: dict[str, Any]) -> None:
    if payload.get("stage") != "random":
        raise ValueError("Stage 1 payload must be random search")
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != STAGE1_TRIAL_COUNT:
        raise ValueError("Stage 1 must contain exactly 40 trials")
    expected_ids = [f"stage1-{index:03d}" for index in range(1, 41)]
    if [trial["trial_id"] for trial in trials] != expected_ids:
        raise ValueError("Stage 1 trial IDs are not frozen in canonical order")
    for trial in trials:
        if trial["optimizer"] not in OPTIMIZERS:
            raise ValueError("Invalid optimizer")
        if not 1e-5 <= float(trial["learning_rate"]) <= 1e-2:
            raise ValueError("Learning rate outside search space")
        if not 8 <= int(trial["batch_size"]) <= 256:
            raise ValueError("Batch size outside search space")
        weight_decay = float(trial["weight_decay"])
        if weight_decay != 0.0 and not 1e-6 <= weight_decay <= 1e-2:
            raise ValueError("Weight decay outside search space")
        if not 0.0 <= float(trial["dropout"]) <= 0.5:
            raise ValueError("Dropout outside search space")
        if not 1 <= int(trial["depth"]) <= 7:
            raise ValueError("Depth outside search space")
        if not 8 <= int(trial["start_filters"]) <= 64:
            raise ValueError("Start filters outside search space")
        expected_parameters = scratch_parameter_count(
            int(trial["depth"]), int(trial["start_filters"])
        )
        if int(trial["parameter_count"]) != expected_parameters:
            raise ValueError("Parameter count does not match architecture")
