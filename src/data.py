from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Sequence
from zipfile import ZipFile

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError


DATASET_ID = "DATA-001"
DATASET_NAME = "original_rgb"
EXPECTED_ARCHIVE_NAME = "images.zip"
EXPECTED_ARCHIVE_SIZE = 939_921_132
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
VALID_LABELS = frozenset({"flip", "notflip"})
VALID_SUPPLIED_SPLITS = frozenset({"training", "testing"})
_IMAGE_NAME_RE = re.compile(r"^(?P<source>\d{4})_(?P<frame>\d{9})\.jpg$")


@dataclass(frozen=True)
class ParsedImagePath:
    archive_member: str
    supplied_split: str
    label: str
    source_id: str
    frame_number: int


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    archive_member: str
    supplied_split: str
    label: str
    source_id: str
    frame_number: int
    width: int
    height: int
    aspect_ratio: float
    original_mode: str
    file_size_bytes: int
    file_sha256: str
    pixel_sha256: str


MANIFEST_FIELDS = tuple(ManifestRow.__dataclass_fields__.keys())


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive_identity(
    path: str | Path,
    *,
    expected_size: int = EXPECTED_ARCHIVE_SIZE,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> dict[str, object]:
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    size = archive.stat().st_size
    digest = sha256_file(archive)
    if size != expected_size:
        raise ValueError(f"Archive size mismatch: {size} != {expected_size}")
    if digest != expected_sha256:
        raise ValueError(f"Archive SHA-256 mismatch: {digest} != {expected_sha256}")
    return {"file_name": archive.name, "size_bytes": size, "sha256": digest}


def parse_archive_member(member: str) -> ParsedImagePath:
    path = PurePosixPath(member)
    parts = path.parts
    if len(parts) != 4 or parts[0] != "images":
        raise ValueError(f"Unexpected image path structure: {member}")
    supplied_split, label, filename = parts[1], parts[2], parts[3]
    if supplied_split not in VALID_SUPPLIED_SPLITS:
        raise ValueError(f"Unexpected supplied split in {member}: {supplied_split}")
    if label not in VALID_LABELS:
        raise ValueError(f"Unexpected label in {member}: {label}")
    match = _IMAGE_NAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected image filename: {member}")
    return ParsedImagePath(
        archive_member=member,
        supplied_split=supplied_split,
        label=label,
        source_id=match.group("source"),
        frame_number=int(match.group("frame")),
    )


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(rgb.width.to_bytes(8, "big", signed=False))
    digest.update(rgb.height.to_bytes(8, "big", signed=False))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _sample_id(parsed: ParsedImagePath) -> str:
    return f"{parsed.supplied_split}/{parsed.label}/{parsed.source_id}_{parsed.frame_number:09d}"


def scan_archive(archive_path: str | Path) -> tuple[list[ManifestRow], dict[str, object]]:
    rows: list[ManifestRow] = []
    split_label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    pixel_hash_members: defaultdict[str, list[ManifestRow]] = defaultdict(list)

    train_pixel_count = 0
    train_sum = np.zeros(3, dtype=np.float64)
    train_sq_sum = np.zeros(3, dtype=np.float64)

    with ZipFile(archive_path) as archive:
        infos = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda x: x.filename)
        if not infos:
            raise ValueError("Archive contains no files")
        seen_members: set[str] = set()
        for info in infos:
            if info.filename in seen_members:
                raise ValueError(f"Duplicate archive member: {info.filename}")
            seen_members.add(info.filename)
            parsed = parse_archive_member(info.filename)
            encoded = archive.read(info.filename)
            file_digest = hashlib.sha256(encoded).hexdigest()
            try:
                with Image.open(BytesIO(encoded)) as image:
                    image.load()
                    width, height = image.size
                    original_mode = image.mode
                    rgb = image.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Image decode failed for {info.filename}: {exc}") from exc

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {info.filename}: {(width, height)}")

            pixel_digest = decoded_pixel_sha256(rgb)
            row = ManifestRow(
                sample_id=_sample_id(parsed),
                archive_member=parsed.archive_member,
                supplied_split=parsed.supplied_split,
                label=parsed.label,
                source_id=parsed.source_id,
                frame_number=parsed.frame_number,
                width=width,
                height=height,
                aspect_ratio=round(width / height, 8),
                original_mode=original_mode,
                file_size_bytes=len(encoded),
                file_sha256=file_digest,
                pixel_sha256=pixel_digest,
            )
            rows.append(row)
            split_label_counts[f"{row.supplied_split}/{row.label}"] += 1
            source_counts[row.source_id] += 1
            dimension_counts[f"{width}x{height}"] += 1
            pixel_hash_members[pixel_digest].append(row)

            if row.supplied_split == "training":
                array = np.asarray(rgb, dtype=np.float64) / 255.0
                flat = array.reshape(-1, 3)
                train_pixel_count += flat.shape[0]
                train_sum += flat.sum(axis=0)
                train_sq_sum += np.square(flat).sum(axis=0)

    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Manifest sample_id values are not unique")
    if {row.label for row in rows} - VALID_LABELS:
        raise ValueError("Manifest contains an invalid label")
    if any(row.frame_number < 0 for row in rows):
        raise ValueError("Manifest contains a negative FrameNumber")

    duplicate_groups = [group for group in pixel_hash_members.values() if len(group) > 1]
    cross_label_duplicate_groups = sum(1 for group in duplicate_groups if len({r.label for r in group}) > 1)
    cross_split_duplicate_groups = sum(1 for group in duplicate_groups if len({r.supplied_split for r in group}) > 1)

    train_mean = train_sum / train_pixel_count
    train_var = np.maximum(train_sq_sum / train_pixel_count - np.square(train_mean), 0.0)
    train_std = np.sqrt(train_var)

    summary: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "image_count": len(rows),
        "label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "supplied_split_counts": dict(sorted(Counter(row.supplied_split for row in rows).items())),
        "split_label_counts": dict(sorted(split_label_counts.items())),
        "source_count": len(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "aspect_ratio_min": min(row.aspect_ratio for row in rows),
        "aspect_ratio_max": max(row.aspect_ratio for row in rows),
        "exact_pixel_duplicate_group_count": len(duplicate_groups),
        "exact_pixel_duplicate_image_count": sum(len(group) for group in duplicate_groups),
        "cross_label_exact_duplicate_group_count": cross_label_duplicate_groups,
        "cross_split_exact_duplicate_group_count": cross_split_duplicate_groups,
        "training_rgb_mean": [round(float(x), 10) for x in train_mean],
        "training_rgb_std": [round(float(x), 10) for x in train_std],
        "training_rgb_pixel_count": train_pixel_count,
    }
    return rows, summary


def write_manifest(rows: Sequence[ManifestRow], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return sha256_file(output)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_manifest(path: str | Path) -> dict[str, object]:
    rows = load_manifest(path)
    return {
        "image_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "supplied_split_counts": dict(sorted(Counter(row["supplied_split"] for row in rows).items())),
        "source_count": len({row["source_id"] for row in rows}),
        "dimension_counts": dict(sorted(Counter(f'{row["width"]}x{row["height"]}' for row in rows).items())),
    }


def select_representative_rows(
    rows: Sequence[ManifestRow], *, per_stratum: int = 2, seed: int = 42
) -> list[ManifestRow]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1")
    grouped: defaultdict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.supplied_split, row.label)].append(row)

    rng = random.Random(seed)
    selected: list[ManifestRow] = []
    for key in sorted(grouped):
        source_groups: defaultdict[str, list[ManifestRow]] = defaultdict(list)
        for row in grouped[key]:
            source_groups[row.source_id].append(row)
        sources = sorted(source_groups)
        chosen_sources = rng.sample(sources, k=min(per_stratum, len(sources)))
        for source in sorted(chosen_sources):
            candidates = sorted(source_groups[source], key=lambda row: row.frame_number)
            selected.append(candidates[len(candidates) // 2])
    return selected


def write_representative_sample(
    archive_path: str | Path,
    rows: Sequence[ManifestRow],
    output_dir: str | Path,
    *,
    per_stratum: int = 2,
    seed: int = 42,
) -> list[ManifestRow]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_representative_rows(rows, per_stratum=per_stratum, seed=seed)
    with ZipFile(archive_path) as archive:
        for row in selected:
            filename = f"{row.supplied_split}_{row.label}_{row.source_id}_{row.frame_number:09d}.jpg"
            (output / filename).write_bytes(archive.read(row.archive_member))
    write_manifest(selected, output / "sample_manifest.csv")
    return selected


def write_dataset_registration(
    output_path: str | Path,
    *,
    archive_identity: dict[str, object],
    manifest_path: str | Path,
    manifest_sha256: str,
    summary: dict[str, object],
) -> None:
    registration = {
        "dataset_id": DATASET_ID,
        "name": DATASET_NAME,
        "status": "canonical",
        "description": "Original RGB MonReader images with supplied flip/notflip labels and supplied training/testing membership.",
        "source_archive": archive_identity,
        "manifest": {
            "path": Path(manifest_path).as_posix(),
            "sha256": manifest_sha256,
            "row_count": summary["image_count"],
        },
        "labels": sorted(VALID_LABELS),
        "supplied_splits": sorted(VALID_SUPPLIED_SPLITS),
        "lineage": {"parent_dataset_id": None, "transform": "none"},
        "training_only_statistics": {
            "rgb_mean": summary["training_rgb_mean"],
            "rgb_std": summary["training_rgb_std"],
            "pixel_count": summary["training_rgb_pixel_count"],
            "population": "supplied training partition only",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(registration, sort_keys=False), encoding="utf-8")


def build_dataset_artifacts(
    archive_path: str | Path,
    output_root: str | Path,
    *,
    expected_size: int = EXPECTED_ARCHIVE_SIZE,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
    representative_per_stratum: int = 2,
) -> dict[str, object]:
    output_root = Path(output_root)
    manifest_dir = output_root / "manifests" / "datasets"
    sample_dir = output_root / "docs" / "sample_images"
    archive_identity = verify_archive_identity(
        archive_path, expected_size=expected_size, expected_sha256=expected_sha256
    )
    rows, summary = scan_archive(archive_path)
    manifest_path = manifest_dir / f"{DATASET_ID}.csv"
    manifest_sha256 = write_manifest(rows, manifest_path)
    write_dataset_registration(
        manifest_dir / f"{DATASET_ID}.yaml",
        archive_identity=archive_identity,
        manifest_path=Path("manifests") / "datasets" / f"{DATASET_ID}.csv",
        manifest_sha256=manifest_sha256,
        summary=summary,
    )
    (manifest_dir / f"{DATASET_ID}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = write_representative_sample(
        archive_path,
        rows,
        sample_dir,
        per_stratum=representative_per_stratum,
        seed=42,
    )
    lineage = {
        "datasets": [
            {
                "dataset_id": DATASET_ID,
                "name": DATASET_NAME,
                "parent_dataset_id": None,
                "transform": "none",
                "manifest": f"manifests/datasets/{DATASET_ID}.csv",
            }
        ]
    }
    lineage_path = output_root / "manifests" / "dataset_lineage.yaml"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(yaml.safe_dump(lineage, sort_keys=False), encoding="utf-8")
    return {
        "archive": archive_identity,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "summary": summary,
        "representative_sample_count": len(selected),
    }
