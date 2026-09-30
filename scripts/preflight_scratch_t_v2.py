from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_T_SHA256 = "0a1f39ebb01f14989aad37b87c39fbb874307f81bdc4770d15e8df7c7b9aa625"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    path = Path("manifests/splits/T.csv")
    if sha256_file(path) != EXPECTED_T_SHA256:
        raise RuntimeError("Frozen T split SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2989:
        raise RuntimeError(f"T must account for 2989 rows, found {len(rows)}")
    counts = Counter(r["role"] for r in rows)
    if counts != Counter({"train": 2365, "test": 624}):
        raise RuntimeError(f"Unexpected T role counts: {counts}")

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
        expected_detail = "earlier_80pct_train" if r["role"] == "train" else "later_20pct_test"
        if r["role_detail"] != expected_detail:
            raise RuntimeError(f"Unexpected T role_detail for {r['sample_id']}")

    if len(by_video) != 117:
        raise RuntimeError(f"T must contain 117 canonical videos, found {len(by_video)}")

    for video_id, group in by_video.items():
        lengths = {int(r["video_observed_length"]) for r in group}
        cuts = {int(r["temporal_cut_floor80"]) for r in group}
        if len(lengths) != 1 or len(cuts) != 1:
            raise RuntimeError(f"Inconsistent temporal metadata for {video_id}")
        n = next(iter(lengths))
        cut = next(iter(cuts))
        if n != len(group) or cut != math.floor(0.80 * n):
            raise RuntimeError(f"Temporal count/cut mismatch for {video_id}")
        ranks = sorted(int(r["temporal_rank_0based"]) for r in group)
        if ranks != list(range(n)):
            raise RuntimeError(f"Temporal ranks are not contiguous for {video_id}")
        train_ranks = [int(r["temporal_rank_0based"]) for r in group if r["role"] == "train"]
        test_ranks = [int(r["temporal_rank_0based"]) for r in group if r["role"] == "test"]
        if not train_ranks or not test_ranks:
            raise RuntimeError(f"T requires train and test rows for every video: {video_id}")
        if max(train_ranks) >= min(test_ranks):
            raise RuntimeError(f"Non-strict temporal boundary for {video_id}")
        if any(rank >= cut for rank in train_ranks) or any(rank < cut for rank in test_ranks):
            raise RuntimeError(f"T cut assignment mismatch for {video_id}")

    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    train_videos = {r["video_id"] for r in train}
    test_videos = {r["video_id"] for r in test}
    if train_videos != test_videos or len(train_videos) != 117:
        raise RuntimeError("T must use the same 117 videos on both sides of the temporal boundary")

    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "train": len(train),
        "test": len(test),
        "videos": len(by_video),
        "train_test_video_overlap": len(train_videos & test_videos),
        "strict_temporal_boundaries": True,
        "training_environments": sorted({r["environment_id"] for r in train}),
        "test_environments": sorted({r["environment_id"] for r in test}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
