from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_S_SHA256 = "a54cce72ad1f275096f06b98bbc9ec32393e9f112cc0536abd132571e1919619"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    path = Path("manifests/splits/S.csv")
    if sha256_file(path) != EXPECTED_S_SHA256:
        raise RuntimeError("Frozen S split SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2989:
        raise RuntimeError(f"S must account for 2989 rows, found {len(rows)}")
    counts = Counter(r["role"] for r in rows)
    if counts != Counter({"train": 2219, "test": 770}):
        raise RuntimeError(f"Unexpected S role counts: {counts}")
    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    if any(r["environment_id"] == "ENV-03" for r in train):
        raise RuntimeError("ENV-03 leaked into S training")
    if any(r["environment_id"] != "ENV-03" for r in test):
        raise RuntimeError("S test contains a non-ENV-03 image")
    train_videos = {r["video_id"] for r in train}
    test_videos = {r["video_id"] for r in test}
    if train_videos & test_videos:
        raise RuntimeError("S train/test video overlap is non-zero")
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "train": len(train),
        "test": len(test),
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "train_test_video_overlap": 0,
        "training_environments": sorted({r["environment_id"] for r in train}),
        "test_environments": sorted({r["environment_id"] for r in test}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
