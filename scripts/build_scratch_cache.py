from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import join_manifest_with_membership
from src.preprocessing import CANVAS_HEIGHT, CANVAS_WIDTH, build_base_transform


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the exact Stage-1 uint8 preprocessing cache.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("manifests/splits/SPLIT-007_original_tuning.csv"),
    )
    parser.add_argument("--output-data", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    args = parser.parse_args()

    rows = join_manifest_with_membership(
        "manifests/datasets/DATA-001.csv",
        args.membership,
        allowed_partitions={"fit", "validation"},
    )
    if len(rows) != 2392:
        raise ValueError(f"Expected 2,392 Stage-1 development samples, found {len(rows)}")
    transform = build_base_transform("manifests/datasets/DATA-001.yaml")
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        args.output_data,
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH),
    )
    index_rows = []
    for index, row in enumerate(rows):
        path = args.image_root / row["archive_member"]
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        byte_tensor = torch.round(tensor * 255.0).to(dtype=torch.uint8)
        reconstructed = byte_tensor.to(dtype=torch.float32) / 255.0
        if not torch.equal(tensor, reconstructed):
            raise RuntimeError(f"Cache quantization changed transformed pixels for {row['sample_id']}")
        cache[index] = byte_tensor.numpy()
        index_rows.append(
            {
                "cache_index": index,
                "sample_id": row["sample_id"],
                "label": row["label"],
                "video_id": row["video_id"],
                "frame_number": row["frame_number"],
                "partition": row["partition"],
            }
        )
    cache.flush()
    del cache

    with args.output_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cache_index",
                "sample_id",
                "label",
                "video_id",
                "frame_number",
                "partition",
            ),
        )
        writer.writeheader()
        writer.writerows(index_rows)

    payload = {
        "row_count": len(rows),
        "shape": [len(rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH],
        "dtype": "uint8",
        "data_sha256": sha256_file(args.output_data),
        "index_sha256": sha256_file(args.output_index),
        "exact_transform_cache": True,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
