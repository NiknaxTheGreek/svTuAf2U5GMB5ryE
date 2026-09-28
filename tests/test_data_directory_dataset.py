from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data import MonReaderDirectoryDataset
from src.preprocessing import BaseImageTransform


def test_directory_dataset_reads_extracted_archive_layout(tmp_path: Path) -> None:
    member = Path("images/training/notflip/0001_000000001.jpg")
    path = tmp_path / member
    path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 30), color=(30, 40, 50)).save(path)
    dataset = MonReaderDirectoryDataset(
        tmp_path,
        [
            {
                "archive_member": member.as_posix(),
                "label": "notflip",
                "sample_id": "training/notflip/0001_000000001",
                "video_id": "notflip/0001",
                "frame_number": "1",
            }
        ],
        BaseImageTransform(width=32, height=48, pad_rgb=(0.5, 0.5, 0.5)),
    )
    item = dataset[0]
    assert item["image"].shape == (3, 48, 32)
    assert item["label"].item() == 0.0
