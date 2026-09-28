from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.models import ScratchCNN, ScratchCNNConfig
from src.training import EarlyStopping, build_optimizer, evaluate_binary_model, train_one_epoch


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        label = float(index % 2)
        return {
            "image": torch.full((3, 32, 32), label, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": f"s{index}",
            "video_id": f"v{index // 2}",
            "frame_number": index,
        }


def test_early_stopping_keeps_earliest_best_tie() -> None:
    stopper = EarlyStopping(patience=2, max_epochs=10)
    assert stopper.step(0.6, 1) == (True, False)
    assert stopper.step(0.6, 2) == (False, False)
    improved, stop = stopper.step(0.5, 3)
    assert not improved and stop
    assert stopper.best_epoch == 1
    assert stopper.best_metric == 0.6


def test_one_epoch_and_evaluation_smoke() -> None:
    model = ScratchCNN(ScratchCNNConfig(depth=1, start_filters=8, dropout=0.0))
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    optimizer = build_optimizer(
        model.parameters(), name="adam", learning_rate=1e-3
    )
    train = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
    )
    evaluation = evaluate_binary_model(
        model, loader, device=torch.device("cpu")
    )
    assert set(train) == {"loss", "f1", "precision", "recall"}
    assert evaluation["metrics"]["n"] == 8
    assert len(evaluation["probabilities"]) == 8
