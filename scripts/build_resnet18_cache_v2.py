from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.resnet18_v2 import EXPECTED_COUNTS
from src.scratch_v2 import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    read_csv,
    sha256_file,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--membership", required=True, type=Path)
    ap.add_argument("--regime", required=True, choices=("O","S","T","ST"))
    ap.add_argument("--role", required=True, choices=("train","test"))
    ap.add_argument("--output-data", required=True, type=Path)
    ap.add_argument("--output-index", required=True, type=Path)
    ap.add_argument("--output-receipt", required=True, type=Path)
    args = ap.parse_args()

    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("Archive byte-size mismatch")
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive SHA mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest SHA mismatch")
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256[args.regime]:
        raise ValueError("Split SHA mismatch")

    manifest = read_csv(args.manifest)
    membership = read_csv(args.membership)
    manifest_by_id = {r["sample_id"]: r for r in manifest}
    selected = [r for r in membership if r["role"] == args.role]
    expected = EXPECTED_COUNTS[(args.regime, args.role)]
    if len(selected) != expected:
        raise ValueError(f"Unexpected {args.regime}/{args.role} count")

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        args.output_data, mode="w+", dtype=np.uint8,
        shape=(len(selected), 3, 224, 224),
    )
    index_rows = []
    with ZipFile(args.archive) as archive:
        for i, member in enumerate(selected):
            source = manifest_by_id[member["sample_id"]]
            with archive.open(source["archive_member"]) as handle:
                with Image.open(handle) as image:
                    image = image.convert("RGB")
                    image = TF.resize(
                        image, 256,
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                    image = TF.center_crop(image, [224, 224])
                    tensor = TF.pil_to_tensor(image)
            if tuple(tensor.shape) != (3,224,224) or tensor.dtype.name if False else False:
                pass
            cache[i] = tensor.numpy()
            index_rows.append({
                "cache_index": i,
                "sample_id": member["sample_id"],
                "label": member["label"],
                "video_id": member["video_id"],
                "frame_number": int(member["frame_number"]),
                "environment_id": member["environment_id"],
                "regime": args.regime,
                "role": args.role,
                "archive_member": source["archive_member"],
            })
            if (i + 1) % 250 == 0 or i + 1 == len(selected):
                print(f"resnet_cache_progress={i+1}/{len(selected)}", flush=True)
    cache.flush()
    del cache

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    with args.output_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    receipt = {
        "regime": args.regime,
        "role": args.role,
        "rows": len(index_rows),
        "shape": [len(index_rows), 3, 224, 224],
        "dtype": "uint8",
        "archive_sha256": sha256_file(args.archive),
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.membership),
        "cache_data_sha256": sha256_file(args.output_data),
        "cache_index_sha256": sha256_file(args.output_index),
        "preprocessing": {
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "resize_shorter_side": 256,
            "center_crop": [224,224],
            "interpolation": "bilinear",
            "antialias": True,
            "normalization_at_model_input": {
                "mean": [0.485,0.456,0.406],
                "std": [0.229,0.224,0.225],
            },
            "augmentation": False,
        },
    }
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
