from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CANVAS_WIDTH = 224
CANVAS_HEIGHT = 398


@dataclass(frozen=True)
class BaseImageTransform:
    width: int = CANVAS_WIDTH
    height: int = CANVAS_HEIGHT
    pad_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if len(self.pad_rgb) != 3 or any(not 0.0 <= value <= 1.0 for value in self.pad_rgb):
            raise ValueError("pad_rgb must contain three values in [0,1]")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        source_width, source_height = rgb.size
        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, min(self.width, round(source_width * scale)))
        resized_height = max(1, min(self.height, round(source_height * scale)))
        resized = TF.resize(
            rgb,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        pad_left = (self.width - resized_width) // 2
        pad_right = self.width - resized_width - pad_left
        pad_top = (self.height - resized_height) // 2
        pad_bottom = self.height - resized_height - pad_top
        fill = tuple(int(round(value * 255.0)) for value in self.pad_rgb)
        canvas = TF.pad(
            resized,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=fill,
            padding_mode="constant",
        )
        tensor = TF.pil_to_tensor(canvas).to(dtype=torch.float32) / 255.0
        if tensor.shape != (3, self.height, self.width):
            raise RuntimeError(f"Unexpected transformed shape: {tuple(tensor.shape)}")
        return tensor


def load_training_rgb_mean(registration_path: str | Path) -> tuple[float, float, float]:
    payload = yaml.safe_load(Path(registration_path).read_text(encoding="utf-8"))
    values = payload["training_only_statistics"]["rgb_mean"]
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError("Dataset registration does not contain a valid RGB mean")
    mean = tuple(float(value) for value in values)
    if any(not 0.0 <= value <= 1.0 for value in mean):
        raise ValueError("Training RGB mean must be in [0,1]")
    return mean


def build_base_transform(
    registration_path: str | Path,
    *,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> BaseImageTransform:
    return BaseImageTransform(
        width=width,
        height=height,
        pad_rgb=load_training_rgb_mean(registration_path),
    )
