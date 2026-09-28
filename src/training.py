from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    params = list(parameters)
    key = name.lower()
    common = {"params": params, "lr": learning_rate, "weight_decay": weight_decay}
    if key == "adam":
        return torch.optim.Adam(**common)
    if key == "adamw":
        return torch.optim.AdamW(**common)
    if key == "sgd":
        return torch.optim.SGD(**common, momentum=momentum)
    if key == "rmsprop":
        return torch.optim.RMSprop(**common, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")
