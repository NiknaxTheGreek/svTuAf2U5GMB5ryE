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

EXPECTED_COUNTS = {
    "heldout_environment_earlier_context": 609,
    "known_environment_later_context": 463,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--role-detail", required=True, choices=tuple(EXPECTED_COUNTS))
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
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("ST split SHA-256 mismatch")

    manifest = read_csv(args.manifest)
    membership = read_csv(args.membership)
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    selected = [
        row for row in membership
        if row["role"] == "context" and row["role_detail"] == args.role_detail
    ]
    expected_n = EXPECTED_COUNTS[args.role_detail]
    if len(selected) != expected_n:
        raise ValueError(f"Unexpected {args.role_detail} size: {len(selected)} != {expected_n}")

    if args.role_detail == "heldout_environment_earlier_context":
        if {row["environment_id"] for row in selected} != {"ENV-03"}:
            raise ValueError("Unseen-early context must be ENV-03 only")
        if any(int(row["temporal_rank_0based"]) >= int(row["temporal_cut_floor80"]) for row in selected):
            raise ValueError("Late row found in unseen-early context")
    else:
        if "ENV-03" in {row["environment_id"] for row in selected}:
            raise ValueError("Known-future context cannot contain ENV-03")
        if any(int(row["temporal_rank_0based"]) < int(row["temporal_cut_floor80"]) for row in selected):
            raise ValueError("Early row found in known-future context")

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        args.output_data, mode="w+", dtype=np.uint8,
        shape=(len(selected), 3, CANVAS_HEIGHT, CANVAS_WIDTH),
    )
    index_rows = []
    with ZipFile(args.archive) as archive:
        for idx, member in enumerate(selected):
            sample_id = member["sample_id"]
            source = manifest_by_id[sample_id]
            for key in ("label", "video_id", "frame_number", "supplied_split"):
                if str(source[key]) != str(member[key]):
                    raise ValueError(f"Manifest/split mismatch for {sample_id} field {key}")
            with archive.open(source["archive_member"]) as handle:
                with Image.open(handle) as image:
                    array = preprocess_to_uint8(image)
            cache[idx] = array
            index_rows.append({
                "cache_index": idx,
                "sample_id": sample_id,
                "label": member["label"],
                "video_id": member["video_id"],
                "frame_number": int(member["frame_number"]),
                "environment_id": member["environment_id"],
                "role_detail": member["role_detail"],
                "archive_member": source["archive_member"],
            })
            if (idx + 1) % 250 == 0 or idx + 1 == len(selected):
                print(f"context_cache_progress={idx + 1}/{len(selected)}", flush=True)
    cache.flush()
    del cache

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    fields = list(index_rows[0])
    with args.output_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    receipt = {
        "regime": "ST",
        "role": "context",
        "role_detail": args.role_detail,
        "rows": len(index_rows),
        "shape": [len(index_rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH],
        "dtype": "uint8",
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
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
