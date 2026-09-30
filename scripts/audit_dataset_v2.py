from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

EXPECTED_ARCHIVE_NAME = "images.zip"
EXPECTED_ARCHIVE_SIZE = 939_921_132
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
VALID_LABELS = {"flip", "notflip"}
VALID_SPLITS = {"training", "testing"}
IMAGE_RE = re.compile(r"^(?P<video>\d{4})_(?P<frame>\d{9})\.jpg$")


@dataclass(frozen=True)
class Row:
    sample_id: str
    archive_member: str
    supplied_split: str
    label: str
    raw_video_id: str
    video_id: str
    frame_number: int
    width: int
    height: int
    original_mode: str
    original_channels: int
    rgb_channels: int
    aspect_ratio: float
    encoded_bytes: int
    encoded_sha256: str
    decoded_rgb_sha256: str
    decode_ok: bool


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(path: Path, expected_size: int, expected_sha256: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = sha256_file(path)
    receipt = {
        "archive_name": path.name,
        "size_bytes": size,
        "sha256": digest,
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha256,
        "size_match": size == expected_size,
        "sha256_match": digest == expected_sha256,
    }
    if not receipt["size_match"] or not receipt["sha256_match"]:
        raise RuntimeError("DATASET IDENTITY FAILURE: " + json.dumps(receipt, sort_keys=True))
    return receipt


def parse_member(member: str) -> tuple[str, str, str, str, int]:
    p = PurePosixPath(member)
    if len(p.parts) != 4 or p.parts[0] != "images":
        raise ValueError(f"Unexpected archive member structure: {member}")
    supplied_split, label, filename = p.parts[1], p.parts[2], p.parts[3]
    if supplied_split not in VALID_SPLITS:
        raise ValueError(f"Unexpected supplied split {supplied_split!r}: {member}")
    if label not in VALID_LABELS:
        raise ValueError(f"Unexpected label {label!r}: {member}")
    m = IMAGE_RE.fullmatch(filename)
    if not m:
        raise ValueError(f"Unexpected filename: {member}")
    raw_video_id = m.group("video")
    frame = int(m.group("frame"))
    video_id = f"{label}/{raw_video_id}"
    return supplied_split, label, raw_video_id, video_id, frame


def pixel_hash_rgb(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    h = hashlib.sha256()
    h.update(b"RGB\\0")
    h.update(rgb.width.to_bytes(8, "big"))
    h.update(rgb.height.to_bytes(8, "big"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def missing_ranges(frames: list[int]) -> tuple[int, int, list[str]]:
    ranges: list[str] = []
    missing = 0
    gaps = 0
    for a, b in zip(frames, frames[1:]):
        if b > a + 1:
            gaps += 1
            missing += b - a - 1
            ranges.append(f"{a + 1}-{b - 1}" if b > a + 2 else str(a + 1))
    return gaps, missing, ranges


def summarize(rows: list[Row]) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    if not rows:
        raise RuntimeError("No image rows were produced")

    labels = Counter(r.label for r in rows)
    splits = Counter(r.supplied_split for r in rows)
    split_labels = Counter((r.supplied_split, r.label) for r in rows)
    modes = Counter(r.original_mode for r in rows)
    dimensions = Counter((r.width, r.height) for r in rows)

    by_video: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_video[r.video_id].append(r)

    sample_ids = [r.sample_id for r in rows]
    members = [r.archive_member for r in rows]
    video_frames = [(r.video_id, r.frame_number) for r in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("sample_id values are not unique")
    if len(members) != len(set(members)):
        raise RuntimeError("archive member names are not unique")
    if len(video_frames) != len(set(video_frames)):
        raise RuntimeError("canonical video_id + frame_number pairs are not unique")

    video_rows: list[dict] = []
    video_lengths: list[int] = []
    for video_id in sorted(by_video):
        group = sorted(by_video[video_id], key=lambda r: r.frame_number)
        frames = [r.frame_number for r in group]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise RuntimeError(f"Non-strict frame order after sorting for {video_id}")
        gaps, missing, ranges = missing_ranges(frames)
        video_lengths.append(len(group))
        video_rows.append({
            "video_id": video_id,
            "label": group[0].label,
            "raw_video_id": group[0].raw_video_id,
            "image_count": len(group),
            "min_frame": min(frames),
            "max_frame": max(frames),
            "gap_count": gaps,
            "missing_frame_number_count": missing,
            "missing_frame_ranges": ";".join(ranges),
            "training_images": sum(r.supplied_split == "training" for r in group),
            "testing_images": sum(r.supplied_split == "testing" for r in group),
        })

    by_pixel: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_pixel[r.decoded_rgb_sha256].append(r)
    dup_groups = [g for g in by_pixel.values() if len(g) > 1]
    duplicate_rows: list[dict] = []
    for idx, g in enumerate(sorted(dup_groups, key=lambda x: x[0].decoded_rgb_sha256), 1):
        duplicate_rows.append({
            "duplicate_group_id": f"DUP-{idx:04d}",
            "decoded_rgb_sha256": g[0].decoded_rgb_sha256,
            "image_count": len(g),
            "labels": ";".join(sorted({r.label for r in g})),
            "supplied_splits": ";".join(sorted({r.supplied_split for r in g})),
            "video_ids": ";".join(sorted({r.video_id for r in g})),
            "members": ";".join(sorted(r.archive_member for r in g)),
            "cross_label": len({r.label for r in g}) > 1,
            "cross_split": len({r.supplied_split for r in g}) > 1,
            "cross_video": len({r.video_id for r in g}) > 1,
        })

    train_by_video: dict[str, list[int]] = defaultdict(list)
    test_rows = [r for r in rows if r.supplied_split == "testing"]
    for r in rows:
        if r.supplied_split == "training":
            train_by_video[r.video_id].append(r.frame_number)
    for v in train_by_video:
        train_by_video[v].sort()

    proximity_rows: list[dict] = []
    for r in sorted(test_rows, key=lambda x: (x.video_id, x.frame_number)):
        train_frames = train_by_video.get(r.video_id, [])
        if train_frames:
            nearest = min(abs(r.frame_number - x) for x in train_frames)
            max_train = max(train_frames)
            later_than_all_train = r.frame_number > max_train
        else:
            nearest = None
            max_train = None
            later_than_all_train = None
        proximity_rows.append({
            "sample_id": r.sample_id,
            "video_id": r.video_id,
            "frame_number": r.frame_number,
            "source_seen_in_training": bool(train_frames),
            "nearest_training_frame_distance": nearest if nearest is not None else "",
            "max_training_frame": max_train if max_train is not None else "",
            "later_than_all_training_frames_same_video": later_than_all_train if later_than_all_train is not None else "",
        })

    with_distance = [x for x in proximity_rows if x["nearest_training_frame_distance"] != ""]
    distances = [int(x["nearest_training_frame_distance"]) for x in with_distance]
    future = [x for x in with_distance if x["later_than_all_training_frames_same_video"] is True]

    train_videos = {r.video_id for r in rows if r.supplied_split == "training"}
    test_videos = {r.video_id for r in rows if r.supplied_split == "testing"}

    summary = {
        "image_count": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "supplied_split_counts": dict(sorted(splits.items())),
        "split_label_counts": {f"{s}/{l}": n for (s, l), n in sorted(split_labels.items())},
        "video_count": len(by_video),
        "train_video_count": len(train_videos),
        "test_video_count": len(test_videos),
        "train_test_video_overlap_count": len(train_videos & test_videos),
        "train_only_video_count": len(train_videos - test_videos),
        "test_only_video_count": len(test_videos - train_videos),
        "video_length": {
            "min": min(video_lengths),
            "median": statistics.median(video_lengths),
            "max": max(video_lengths),
        },
        "original_mode_counts": dict(sorted(modes.items())),
        "resolution_counts": {f"{w}x{h}": n for (w, h), n in sorted(dimensions.items())},
        "aspect_ratio": {
            "min": min(r.aspect_ratio for r in rows),
            "median": statistics.median(r.aspect_ratio for r in rows),
            "max": max(r.aspect_ratio for r in rows),
        },
        "decode_failures": sum(not r.decode_ok for r in rows),
        "exact_duplicate": {
            "group_count": len(dup_groups),
            "images_in_duplicate_groups": sum(len(g) for g in dup_groups),
            "cross_split_group_count": sum(len({r.supplied_split for r in g}) > 1 for g in dup_groups),
            "cross_label_group_count": sum(len({r.label for r in g}) > 1 for g in dup_groups),
            "cross_video_group_count": sum(len({r.video_id for r in g}) > 1 for g in dup_groups),
        },
        "temporal_structure": {
            "videos_with_frame_number_gaps": sum(int(v["gap_count"]) > 0 for v in video_rows),
            "total_gap_segments": sum(int(v["gap_count"]) for v in video_rows),
            "total_missing_frame_numbers_within_observed_ranges": sum(int(v["missing_frame_number_count"]) for v in video_rows),
        },
        "original_supplied_split_overlap": {
            "test_images": len(test_rows),
            "test_images_from_training_seen_videos": len(with_distance),
            "test_images_from_unseen_videos": len(test_rows) - len(with_distance),
            "seen_source_fraction": len(with_distance) / len(test_rows) if test_rows else None,
            "nearest_training_frame_distance": {
                "min": min(distances) if distances else None,
                "median": statistics.median(distances) if distances else None,
                "max": max(distances) if distances else None,
                "distance_eq_1_count": sum(d == 1 for d in distances),
                "distance_eq_1_fraction": (sum(d == 1 for d in distances) / len(distances)) if distances else None,
                "distance_le_2_count": sum(d <= 2 for d in distances),
                "distance_le_2_fraction": (sum(d <= 2 for d in distances) / len(distances)) if distances else None,
            },
            "temporally_future_test_images": len(future),
            "temporally_future_test_videos": len({x["video_id"] for x in future}),
        },
        "accounting": {
            "parsed_image_rows": len(rows),
            "unknown_or_skipped_members": 0,
            "total_accounted": len(rows),
        },
    }

    resolution_rows = [
        {"width": w, "height": h, "image_count": n, "aspect_ratio": round(w / h, 8)}
        for (w, h), n in sorted(dimensions.items())
    ]
    return summary, video_rows, duplicate_rows, resolution_rows, proximity_rows


def scan(archive_path: Path) -> list[Row]:
    rows: list[Row] = []
    try:
        with ZipFile(archive_path) as zf:
            infos = sorted((i for i in zf.infolist() if not i.is_dir()), key=lambda i: i.filename)
            if not infos:
                raise RuntimeError("Archive contains no files")
            seen = set()
            for info in infos:
                if info.filename in seen:
                    raise RuntimeError(f"Duplicate archive member name: {info.filename}")
                seen.add(info.filename)
                supplied_split, label, raw_video_id, video_id, frame = parse_member(info.filename)
                try:
                    encoded = zf.read(info.filename)
                except Exception as exc:
                    raise RuntimeError(f"Could not read {info.filename}: {exc}") from exc
                encoded_sha = hashlib.sha256(encoded).hexdigest()
                try:
                    with Image.open(BytesIO(encoded)) as im:
                        im.load()
                        width, height = im.size
                        mode = im.mode
                        channels = len(im.getbands())
                        rgb_hash = pixel_hash_rgb(im)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise RuntimeError(f"Image decode failed for {info.filename}: {exc}") from exc
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"Invalid dimensions for {info.filename}: {width}x{height}")
                rows.append(Row(
                    sample_id=f"{supplied_split}/{video_id}_{frame:09d}",
                    archive_member=info.filename,
                    supplied_split=supplied_split,
                    label=label,
                    raw_video_id=raw_video_id,
                    video_id=video_id,
                    frame_number=frame,
                    width=width,
                    height=height,
                    original_mode=mode,
                    original_channels=channels,
                    rgb_channels=3,
                    aspect_ratio=round(width / height, 10),
                    encoded_bytes=len(encoded),
                    encoded_sha256=encoded_sha,
                    decoded_rgb_sha256=rgb_hash,
                    decode_ok=True,
                ))
    except BadZipFile as exc:
        raise RuntimeError(f"Invalid ZIP archive: {exc}") from exc
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-size", type=int, default=EXPECTED_ARCHIVE_SIZE)
    ap.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    args = ap.parse_args()

    archive = Path(args.archive)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    receipt = verify_archive(archive, args.expected_size, args.expected_sha256)
    (out / "archive_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    rows = scan(archive)
    summary, video_rows, dup_rows, resolution_rows, proximity_rows = summarize(rows)

    manifest_fields = list(Row.__dataclass_fields__.keys())
    write_csv(out / "master_manifest.csv", (asdict(r) for r in rows), manifest_fields)
    write_csv(out / "video_summary.csv", video_rows, list(video_rows[0].keys()))
    write_csv(
        out / "exact_duplicate_groups.csv",
        dup_rows,
        ["duplicate_group_id", "decoded_rgb_sha256", "image_count", "labels", "supplied_splits", "video_ids", "members", "cross_label", "cross_split", "cross_video"],
    )
    write_csv(out / "resolution_summary.csv", resolution_rows, list(resolution_rows[0].keys()))
    write_csv(out / "original_test_nearest_train_frame.csv", proximity_rows, list(proximity_rows[0].keys()))

    summary["archive"] = receipt
    summary["artifact_sha256"] = {
        name: sha256_file(out / name)
        for name in [
            "archive_receipt.json",
            "master_manifest.csv",
            "video_summary.csv",
            "exact_duplicate_groups.csv",
            "resolution_summary.csv",
            "original_test_nearest_train_frame.csv",
        ]
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
