from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.scratch_v2 import (
    MemmapDataset,
    ScratchCNN,
    ScratchCNNConfig,
    build_optimizer,
    candidate_config,
    count_trainable_parameters,
    load_candidate_bank,
    preprocess_to_uint8,
    seed_everything,
)


def main() -> int:
    bank = load_candidate_bank("configs/MONREADER_FIXED_20_CONFIG_BANK.json")
    expected_ids = [f"C{i:02d}" for i in range(1, 21)]
    observed_ids = [row["candidate_id"] for row in bank["candidates"]]
    assert observed_ids == expected_ids

    # All 20 architectures must construct, run forward/backward, and build their optimizer.
    for cid in expected_ids:
        cfg = candidate_config(bank, cid)
        seed_everything(42)
        model = ScratchCNN(ScratchCNNConfig(cfg["depth"], cfg["start_filters"], cfg["dropout"]))
        x = torch.rand(2, 3, 398, 224)
        y = torch.tensor([0.0, 1.0])
        logits = model(x)
        assert logits.shape == (2,)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        optimizer = build_optimizer(model, cfg)
        optimizer.step()
        assert count_trainable_parameters(model) > 0

    # Exact preprocessing/cache round trip on synthetic images.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "cache.npy"
        index_path = root / "index.csv"
        arr = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.uint8, shape=(2, 3, 398, 224))
        for i in range(2):
            image = Image.fromarray(np.full((1920,1080,3), 80 + i * 80, dtype=np.uint8), mode="RGB")
            arr[i] = preprocess_to_uint8(image)
        arr.flush()
        del arr
        index_path.write_text(
            "cache_index,sample_id,label,video_id,frame_number\n"
            "0,s0,notflip,notflip/0001,1\n"
            "1,s1,flip,flip/0001,1\n",
            encoding="utf-8",
        )
        ds = MemmapDataset(data_path, index_path)
        assert len(ds) == 2
        assert ds[0]["image"].shape == (3,398,224)
        assert float(ds[0]["image"].min()) >= 0.0 and float(ds[0]["image"].max()) <= 1.0

    print(json.dumps({
        "status": "PASS",
        "candidate_architectures_checked": 20,
        "forward_backward_checked": 20,
        "cache_roundtrip": True,
        "test_data_used": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
