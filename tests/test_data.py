from __future__ import annotations

import csv
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from src.data import (
    DATASET_ID,
    build_dataset_artifacts,
    parse_archive_member,
    scan_archive,
    sha256_file,
    verify_archive_identity,
)


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> bytes:
    image = Image.new("RGB", size=size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_parse_archive_member() -> None:
    parsed = parse_archive_member("images/training/flip/0058_000000030.jpg")
    assert parsed.supplied_split == "training"
    assert parsed.label == "flip"
    assert parsed.source_id == "0058"
    assert parsed.frame_number == 30


@pytest.mark.parametrize(
    "member",
    [
        "training/flip/0058_000000030.jpg",
        "images/validation/flip/0058_000000030.jpg",
        "images/training/other/0058_000000030.jpg",
        "images/training/flip/not_a_frame.jpg",
        "images/training/flip/0058_30.png",
    ],
)
def test_parse_archive_member_rejects_invalid_structure(member: str) -> None:
    with pytest.raises(ValueError):
        parse_archive_member(member)


def test_verify_archive_identity(tmp_path: Path) -> None:
    archive = tmp_path / "images.zip"
    archive.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    observed = verify_archive_identity(archive, expected_size=3, expected_sha256=digest)
    assert observed["size_bytes"] == 3
    assert observed["sha256"] == digest
    with pytest.raises(ValueError):
        verify_archive_identity(archive, expected_size=4, expected_sha256=digest)


def test_scan_archive_decodes_and_detects_exact_pixel_duplicates(tmp_path: Path) -> None:
    archive = tmp_path / "images.zip"
    same = _jpeg_bytes((10, 20, 30))
    different = _jpeg_bytes((200, 50, 10))
    _write_zip(
        archive,
        {
            "images/training/flip/0001_000000001.jpg": same,
            "images/training/notflip/0002_000000002.jpg": different,
            "images/testing/flip/0003_000000003.jpg": same,
            "images/testing/notflip/0004_000000004.jpg": _jpeg_bytes((100, 100, 100)),
        },
    )
    rows, summary = scan_archive(archive)
    assert len(rows) == 4
    assert summary["source_count"] == 4
    assert summary["exact_pixel_duplicate_group_count"] == 1
    assert summary["cross_split_exact_duplicate_group_count"] == 1
    assert rows[0].width == 8
    assert rows[0].height == 6
    assert rows[0].file_sha256
    assert rows[0].pixel_sha256


def test_build_dataset_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "images.zip"
    _write_zip(
        archive,
        {
            "images/training/flip/0001_000000001.jpg": _jpeg_bytes((10, 20, 30)),
            "images/training/notflip/0002_000000002.jpg": _jpeg_bytes((50, 60, 70)),
            "images/testing/flip/0003_000000003.jpg": _jpeg_bytes((80, 90, 100)),
            "images/testing/notflip/0004_000000004.jpg": _jpeg_bytes((110, 120, 130)),
        },
    )
    output = tmp_path / "generated"
    result = build_dataset_artifacts(
        archive,
        output,
        expected_size=archive.stat().st_size,
        expected_sha256=sha256_file(archive),
        representative_per_stratum=1,
    )
    manifest = output / "manifests" / "datasets" / f"{DATASET_ID}.csv"
    registration = output / "manifests" / "datasets" / f"{DATASET_ID}.yaml"
    summary = output / "manifests" / "datasets" / f"{DATASET_ID}_summary.json"
    sample_manifest = output / "docs" / "sample_images" / "sample_manifest.csv"
    assert result["representative_sample_count"] == 4
    assert manifest.is_file()
    assert registration.is_file()
    assert summary.is_file()
    assert sample_manifest.is_file()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["label"] for row in rows} == {"flip", "notflip"}


def test_scan_archive_rejects_corrupt_jpeg(tmp_path: Path) -> None:
    archive = tmp_path / "images.zip"
    _write_zip(archive, {"images/training/flip/0001_000000001.jpg": b"not a jpeg"})
    with pytest.raises(ValueError, match="Image decode failed"):
        scan_archive(archive)
