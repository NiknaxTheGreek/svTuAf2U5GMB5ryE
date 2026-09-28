from __future__ import annotations

import torch
import yaml
from PIL import Image

from src.preprocessing import BaseImageTransform, build_base_transform


def test_base_transform_produces_fixed_portrait_tensor() -> None:
    transform = BaseImageTransform(width=224, height=398, pad_rgb=(0.2, 0.4, 0.6))
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    tensor = transform(image)
    assert tensor.shape == (3, 398, 224)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    expected = torch.tensor([0.2, 0.4, 0.6])
    assert torch.allclose(tensor[:, 0, 0], expected, atol=1 / 255 + 1e-6)


def test_build_transform_uses_training_mean(tmp_path) -> None:
    registration = tmp_path / "dataset.yaml"
    registration.write_text(
        yaml.safe_dump(
            {
                "training_only_statistics": {
                    "rgb_mean": [0.11, 0.22, 0.33],
                }
            }
        ),
        encoding="utf-8",
    )
    transform = build_base_transform(registration)
    assert transform.pad_rgb == (0.11, 0.22, 0.33)
