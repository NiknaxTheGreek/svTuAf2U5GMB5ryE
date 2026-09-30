from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.scratch_v2 import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    preprocess_to_uint8,
    read_csv,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--regime", required=True, choices=("O", "S", "T", "ST"))
    parser.add_argument("--role", required=True, choices=("train", "test"))
    parser.add_argument("--output-data", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    args = parser.parse_args()

    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("Archive byte-size mismatch")
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest SHA-256 mismatch")
    expected_split = EXPECTED_SPLIT_SHA256[args.regime]
    if sha256_file(args.membership) != expected_split:
        raise ValueError(f"{args.regime} split SHA-256 mismatch")

    manifest = read_csv(args.manifest)
    membership = read_csv(args.membership)
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    selected = [row for row in membership if row["role"] == args.role]
    if not selected:
        raise ValueError("Selected cache population is empty")

    expected_counts = {
        ("O", "train"): 2392, ("O", "test"): 597,
        ("S", "train"): 2219, ("S", "test"): 770,
        ("T", "train"): 2365, ("T", "test"): 624,
        ("ST", "train"): 1756, ("ST", "test"): 161,
    }
    expected_n = expected_counts[(args.regime, args.role)]
    if len(selected) != expected_n:
        raise ValueError(f"Unexpected {args.regime}/{args.role} size: {len(selected)} != {expected_n}")

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)

    cache = np.lib.format.open_memmap(
        args.output_data,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected), 3, CANVAS_HEIGHT, CANVAS_WIDTH),
    )
    index_rows = []
    with ZipFile(args.archive) as archive:
        for index, member in enumerate(selected):
            sample_id = member["sample_id"]
            if sample_id not in manifest_by_id:
                raise ValueError(f"Split sample missing from dataset manifest: {sample_id}")
            source = manifest_by_id[sample_id]
            for key in ("label", "video_id", "frame_number", "supplied_split"):
                if str(source[key]) != str(member[key]):
                    raise ValueError(f"Manifest/split mismatch for {sample_id} field {key}")
            with archive.open(source["archive_member"]) as handle:
                with Image.open(handle) as image:
                    array = preprocess_to_uint8(image)
            cache[index] = array
            index_rows.append(
                {
                    "cache_index": index,
                    "sample_id": sample_id,
                    "label": member["label"],
                    "video_id": member["video_id"],
                    "frame_number": int(member["frame_number"]),
                    "environment_id": member["environment_id"],
                    "supplied_split": member["supplied_split"],
                    "regime": args.regime,
                    "role": args.role,
                    "archive_member": source["archive_member"],
                }
            )
            if (index + 1) % 250 == 0 or index + 1 == len(selected):
                print(f"cache_progress={index + 1}/{len(selected)}", flush=True)

    cache.flush()
    del cache

    fields = list(index_rows[0])
    with args.output_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    receipt = {
        "regime": args.regime,
        "role": args.role,
        "rows": len(index_rows),
        "shape": [len(index_rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH],
        "dtype": "uint8",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.membership),
        "cache_data_sha256": sha256_file(args.output_data),
        "cache_index_sha256": sha256_file(args.output_index),
        "preprocessing": {
            "canvas_hxw": [CANVAS_HEIGHT, CANVAS_WIDTH],
            "aspect_ratio_preserved": True,
            "interpolation": "bilinear_antialias",
            "padding": "centered_constant_black",
            "scaling_at_model_input": "[0,1]",
            "augmentation": False,
        },
    }
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
