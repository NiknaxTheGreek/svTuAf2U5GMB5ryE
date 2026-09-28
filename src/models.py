from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScratchCNNConfig:
    depth: int
    start_filters: int
    dropout: float

    def __post_init__(self) -> None:
        if not 1 <= self.depth <= 7:
            raise ValueError("depth must be between 1 and 7")
        if not 8 <= self.start_filters <= 64:
            raise ValueError("start_filters must be between 8 and 64")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be between 0 and 0.5")


class ScratchCNN(nn.Module):
    def __init__(self, config: ScratchCNNConfig) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_channels = 3
        out_channels = config.start_filters
        for _ in range(config.depth):
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            out_channels *= 2
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(in_channels, 1)
        self.apply(_kaiming_initialize)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.features(inputs)
        values = self.pool(values).flatten(1)
        values = self.dropout(values)
        return self.classifier(values)


def _kaiming_initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
