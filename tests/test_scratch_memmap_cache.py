from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import MonReaderMemmapDataset
from src.preprocessing import BaseImageTransform


def test_memmap_dataset_preserves_cached_tensor_exactly(tmp_path: Path) -> None:
    transform = BaseImageTransform(width=224, height=398, pad_rgb=(0.2, 0.3, 0.4))
    image = Image.new("RGB", (108, 192), color=(17, 83, 201))
    expected = transform(image)
    byte_tensor = torch.round(expected * 255.0).to(torch.uint8)
    assert torch.equal(expected, byte_tensor.float() / 255.0)

    data = tmp_path / "cache.npy"
    array = np.lib.format.open_memmap(
        data, mode="w+", dtype=np.uint8, shape=(1, 3, 398, 224)
    )
    array[0] = byte_tensor.numpy()
    array.flush()
    del array

    index = tmp_path / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cache_index", "sample_id", "label", "video_id", "frame_number", "partition"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "cache_index": 0,
                "sample_id": "training/flip/0001_000000001",
                "label": "flip",
                "video_id": "flip/0001",
                "frame_number": 1,
                "partition": "fit",
            }
        )

    rows = [
        {
            "sample_id": "training/flip/0001_000000001",
            "label": "flip",
            "video_id": "flip/0001",
            "frame_number": "1",
        }
    ]
    dataset = MonReaderMemmapDataset(data, index, rows)
    sample = dataset[0]
    assert torch.equal(sample["image"], expected)
    assert sample["label"].item() == 1.0
