from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_ST_SHA256 = "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    path = Path("manifests/splits/ST.csv")
    if sha256_file(path) != EXPECTED_ST_SHA256:
        raise RuntimeError("Frozen ST split SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2989:
        raise RuntimeError(f"ST must account for 2989 rows, found {len(rows)}")
    counts = Counter(r["role"] for r in rows)
    expected_counts = Counter({"train": 1756, "context": 1072, "test": 161})
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected ST role counts: {counts}")

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    if len(by_video) != 117:
        raise RuntimeError(f"ST must contain 117 canonical videos, found {len(by_video)}")

    train = [r for r in rows if r["role"] == "train"]
    test = [r for r in rows if r["role"] == "test"]
    context = [r for r in rows if r["role"] == "context"]

    if {r["environment_id"] for r in train} != {"ENV-01", "ENV-02", "ENV-04"}:
        raise RuntimeError("ST training must use only known environments ENV-01/02/04")
    if {r["environment_id"] for r in test} != {"ENV-03"}:
        raise RuntimeError("ST primary test must contain only held-out ENV-03")
    if any(r["environment_id"] == "ENV-03" for r in train):
        raise RuntimeError("ENV-03 leaked into ST training")

    train_videos = {r["video_id"] for r in train}
    test_videos = {r["video_id"] for r in test}
    if train_videos & test_videos:
        raise RuntimeError("ST train/test canonical-video overlap is non-zero")
    if len(train_videos) != 88 or len(test_videos) != 29:
        raise RuntimeError("Unexpected ST train/test video counts")

    earlier_heldout = [r for r in context if r["role_detail"] == "heldout_environment_earlier_context"]
    later_known = [r for r in context if r["role_detail"] == "known_environment_later_context"]
    if len(earlier_heldout) != 609 or len(later_known) != 463:
        raise RuntimeError("ST context decomposition changed")
    if len(earlier_heldout) + len(later_known) != len(context):
        raise RuntimeError("Unexpected ST context role_detail")

    for r in train:
        if r["role_detail"] != "known_environment_earlier_train":
            raise RuntimeError(f"Unexpected ST train role_detail for {r['sample_id']}")
        if int(r["temporal_rank_0based"]) >= int(r["temporal_cut_floor80"]):
            raise RuntimeError(f"Late known-source row leaked into ST train: {r['sample_id']}")
    for r in test:
        if r["role_detail"] != "heldout_environment_later_test":
            raise RuntimeError(f"Unexpected ST test role_detail for {r['sample_id']}")
        if int(r["temporal_rank_0based"]) < int(r["temporal_cut_floor80"]):
            raise RuntimeError(f"Early heldout row leaked into ST test: {r['sample_id']}")
    for r in earlier_heldout:
        if r["environment_id"] != "ENV-03" or int(r["temporal_rank_0based"]) >= int(r["temporal_cut_floor80"]):
            raise RuntimeError(f"Invalid earlier-heldout context row: {r['sample_id']}")
    for r in later_known:
        if r["environment_id"] == "ENV-03" or int(r["temporal_rank_0based"]) < int(r["temporal_cut_floor80"]):
            raise RuntimeError(f"Invalid later-known context row: {r['sample_id']}")

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

    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "train": len(train),
        "test": len(test),
        "context": len(context),
        "context_earlier_heldout": len(earlier_heldout),
        "context_later_known": len(later_known),
        "train_videos": len(train_videos),
        "test_videos": len(test_videos),
        "train_test_video_overlap": len(train_videos & test_videos),
        "training_environments": sorted({r["environment_id"] for r in train}),
        "test_environments": sorted({r["environment_id"] for r in test}),
        "strict_source_and_temporal_gates": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
