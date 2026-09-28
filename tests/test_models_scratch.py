from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import ScratchCNN, ScratchCNNConfig, count_trainable_parameters


def test_scratch_cnn_structure_and_output() -> None:
    config = ScratchCNNConfig(depth=3, start_filters=8, dropout=0.2)
    model = ScratchCNN(config)
    output = model(torch.zeros(2, 3, 224, 128))
    assert output.shape == (2, 1)
    convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    pools = [module for module in model.modules() if isinstance(module, nn.MaxPool2d)]
    assert len(convs) == 3
    assert len(pools) == 3
    assert [layer.out_channels for layer in convs] == [8, 16, 32]
    assert count_trainable_parameters(model) > 0
    assert all(torch.count_nonzero(layer.weight).item() > 0 for layer in convs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0, "start_filters": 8, "dropout": 0.0},
        {"depth": 8, "start_filters": 8, "dropout": 0.0},
        {"depth": 2, "start_filters": 7, "dropout": 0.0},
        {"depth": 2, "start_filters": 65, "dropout": 0.0},
        {"depth": 2, "start_filters": 8, "dropout": 0.6},
    ],
)
def test_scratch_config_rejects_out_of_range(kwargs) -> None:
    with pytest.raises(ValueError):
        ScratchCNNConfig(**kwargs)
