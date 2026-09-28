import random

import numpy as np
import pytest
import torch

from src.utils import resolve_device, seed_everything


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_everything_is_repeatable() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    second = _draw()
    assert first == second


def test_auto_device_is_cuda_or_cpu() -> None:
    assert resolve_device("auto").type in {"cuda", "cpu"}


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("mps")
