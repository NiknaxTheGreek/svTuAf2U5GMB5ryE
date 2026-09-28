import pytest
import torch

from src.training import build_optimizer


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop"])
def test_supported_optimizers(name: str) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(
        model.parameters(),
        name=name,
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.9,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_unknown_optimizer_is_rejected() -> None:
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        build_optimizer(model.parameters(), name="bad", learning_rate=1e-3)
