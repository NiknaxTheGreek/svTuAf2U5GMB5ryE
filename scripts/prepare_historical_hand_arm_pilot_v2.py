from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from src.scratch_v2 import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    read_csv,
    sha256_file,
)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",required=True,type=Path)
    ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--st-membership",required=True,type=Path)
    ap.add_argument("--sample",required=True,type=Path)
    ap.add_argument("--extract-root",required=True,type=Path)
    ap.add_argument("--inventory",required=True,type=Path)
    ap.add_argument("--receipt",required=True,type=Path)
    args=ap.parse_args()

    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("archive bytes mismatch")
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive SHA mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("manifest SHA mismatch")
    if sha256_file(args.st_membership) != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("ST split SHA mismatch")

    manifest=read_csv(args.manifest)
    by_id={r["sample_id"]:r for r in manifest}
    sample=read_csv(args.sample)
    if len(sample)!=48:
        raise ValueError(f"Expected exact 48-image development sample, got {len(sample)}")

    st=read_csv(args.st_membership)
    context={r["sample_id"] for r in st if r["role"]=="context"}
    test={r["sample_id"] for r in st if r["role"]=="test"}
    ids=[r["sample_id"] for r in sample]
    if len(set(ids))!=48:
        raise ValueError("Duplicate sample IDs in development sample")
    if any(i not in context for i in ids):
        raise ValueError("Development sample is not entirely ST context-only")
    if set(ids)&test:
        raise ValueError("ST primary-test overlap detected")

    args.extract_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ZipFile(args.archive) as z:
        for sid in ids:
            src=by_id[sid]
            member=src["archive_member"]
            dst=args.extract_root/member
            dst.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as fh, dst.open("wb") as out:
                out.write(fh.read())
            rows.append({
                "path":member,
                "label":src["label"],
                "video_id":src["video_id"],
                "frame_num":src["frame_number"],
                "split":src["supplied_split"],
            })

    args.inventory.parent.mkdir(parents=True,exist_ok=True)
    with args.inventory.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","label","video_id","frame_num","split"])
        w.writeheader(); w.writerows(rows)

    receipt={
        "status":"PASS_HISTORICAL_PILOT_INPUT_GATE",
        "images":48,
        "st_context_only":True,
        "primary_test_overlap":0,
        "archive_sha256":sha256_file(args.archive),
        "manifest_sha256":sha256_file(args.manifest),
        "st_split_sha256":sha256_file(args.st_membership),
        "sample_csv_sha256":sha256_file(args.sample),
        "inventory_sha256":sha256_file(args.inventory),
    }
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
