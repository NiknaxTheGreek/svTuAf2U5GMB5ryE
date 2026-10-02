from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.chroma_counterfactual_v2 import rgb_to_ycc709

STATS_STRIDE = 16


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def chroma_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, cb, cr = rgb_to_ycc709(rgb)
    c = np.stack(
        [cb[::STATS_STRIDE, ::STATS_STRIDE], cr[::STATS_STRIDE, ::STATS_STRIDE]],
        axis=-1,
    ).reshape(-1, 2) / 128.0
    return c.mean(axis=0), np.cov(c, rowvar=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split-o", type=Path, required=True)
    ap.add_argument("--split-s", type=Path, required=True)
    ap.add_argument("--split-t", type=Path, required=True)
    ap.add_argument("--split-st", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-summary", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    splits = {
        "O": read_csv(args.split_o),
        "S": read_csv(args.split_s),
        "T": read_csv(args.split_t),
        "ST": read_csv(args.split_st),
    }
    role_by_regime = {
        regime: {r["sample_id"]: r["role"] for r in rows}
        for regime, rows in splits.items()
    }
    env_by_id: dict[str, str] = {}
    for rows in splits.values():
        for r in rows:
            env_by_id.setdefault(r["sample_id"], r["environment_id"])

    donor_ids = sorted(
        sid for sid in by_id
        if all(
            role_by_regime[regime].get(sid) != "test"
            for regime in ("O", "S", "T", "ST")
        )
    )
    if len(donor_ids) != 1428:
        raise ValueError(f"Universal donor pool changed: {len(donor_ids)} != 1428")

    records = []
    with ZipFile(args.archive) as z:
        for i, sid in enumerate(donor_ids, 1):
            m = by_id[sid]
            rgb = decode_member(z, m["archive_member"])
            mu, cov = chroma_stats(rgb)
            records.append({
                "sample_id": sid,
                "video_id": m["video_id"],
                "environment_id": env_by_id[sid],
                "label_for_audit_only": m["label"],
                "mu_cb": float(mu[0]),
                "mu_cr": float(mu[1]),
                "cov_cb_cb": float(cov[0, 0]),
                "cov_cb_cr": float(cov[0, 1]),
                "cov_cr_cr": float(cov[1, 1]),
            })
            if i % 100 == 0 or i == len(donor_ids):
                print(f"donor_stats_progress={i}/{len(donor_ids)}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    summary = {
        "status": "FROZEN_CHROMA_DONOR_STATS",
        "donor_pool_size": len(records),
        "stats_stride": STATS_STRIDE,
        "selection_bearing": False,
        "labels_used_for_stats_or_selection": False,
        "definition": "images non-test simultaneously in O, S, T and ST",
    }
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
