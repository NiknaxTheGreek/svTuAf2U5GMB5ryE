from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


OptimizerName = Literal["adam", "adamw", "sgd", "rmsprop"]
DeviceChoice = Literal["auto", "cpu", "cuda"]
RunMode = Literal["synthetic_smoke"]


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: OptimizerName = "adam"
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: DeviceChoice = "auto"
    mode: RunMode = "synthetic_smoke"
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    project_name: str
    run_name: str
    dataset_id: str
    split_id: str
    seed: int
    device: DeviceChoice
    mode: RunMode
    optimizer_name: OptimizerName
    learning_rate: float
    weight_decay: float
    momentum: float


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return ExperimentConfig.model_validate(payload)


def apply_overrides(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
    device: DeviceChoice | None = None,
) -> ExperimentConfig:
    updates: dict[str, object] = {}
    if seed is not None:
        updates["seed"] = seed
    if device is not None:
        updates["device"] = device
    payload = config.model_dump()
    payload.update(updates)
    return ExperimentConfig.model_validate(payload)


def to_runtime(config: ExperimentConfig) -> RuntimeConfig:
    return RuntimeConfig(
        project_name=config.project_name,
        run_name=config.run_name,
        dataset_id=config.dataset_id,
        split_id=config.split_id,
        seed=config.seed,
        device=config.device,
        mode=config.mode,
        optimizer_name=config.optimizer.name,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        momentum=config.optimizer.momentum,
    )


__all__ = [
    "DeviceChoice",
    "ExperimentConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "ValidationError",
    "apply_overrides",
    "load_config",
    "to_runtime",
]
