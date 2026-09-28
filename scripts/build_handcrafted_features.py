from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data import load_manifest, sha256_file, verify_archive_identity
from src.features import ALL_FEATURES, FEATURE_SIZE, extract_feature_rows


META_FIELDS = (
    "sample_id",
    "label",
    "video_id",
    "frame_number",
    "supplied_split",
    "pixel_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MonReader handcrafted-feature table.")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/datasets/DATA-001.csv")
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity = verify_archive_identity(args.archive)
    manifest_rows = load_manifest(args.manifest)
    feature_rows = extract_feature_rows(
        str(args.archive), manifest_rows, verify_pixel_hashes=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_FIELDS + ALL_FEATURES)
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"archive_sha256={identity['sha256']}")
    print(f"rows={len(feature_rows)}")
    print(f"feature_count={len(ALL_FEATURES)}")
    print(f"feature_size={FEATURE_SIZE[0]}x{FEATURE_SIZE[1]}")
    print(f"feature_csv_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
