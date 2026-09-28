from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from PIL import Image

from src.data import MonReaderZipDataset
from src.preprocessing import BaseImageTransform


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 30), color=(20, 30, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_zip_dataset_returns_image_label_and_metadata(tmp_path) -> None:
    archive = tmp_path / "images.zip"
    member = "images/training/flip/0001_000000001.jpg"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(member, _jpeg_bytes())
    rows = [
        {
            "archive_member": member,
            "label": "flip",
            "sample_id": "training/flip/0001_000000001",
            "video_id": "flip/0001",
            "frame_number": "1",
        }
    ]
    dataset = MonReaderZipDataset(
        archive,
        rows,
        BaseImageTransform(width=32, height=48, pad_rgb=(0.5, 0.5, 0.5)),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 48, 32)
    assert sample["label"].dtype == torch.float32
    assert sample["label"].item() == 1.0
    assert sample["video_id"] == "flip/0001"
    assert sample["frame_number"] == 1
