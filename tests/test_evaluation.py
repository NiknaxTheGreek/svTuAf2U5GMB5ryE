from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation import compute_binary_metrics, label_to_int


def test_metrics_match_sklearn_references() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.9, 0.6, 0.4, 0.2])
    metrics = compute_binary_metrics(y, p)
    predicted = (p >= 0.5).astype(int)
    assert metrics["f1"] == pytest.approx(
        f1_score(y, predicted)
    )
    assert metrics["roc_auc"] == pytest.approx(
        roc_auc_score(y, p)
    )
    assert metrics["confusion_matrix"] == {
        "tn": 2,
        "fp": 1,
        "fn": 1,
        "tp": 2,
    }


def test_threshold_is_fixed_at_half_by_default() -> None:
    y = [0, 1]
    p = [0.499, 0.5]
    metrics = compute_binary_metrics(y, p)
    assert metrics["confusion_matrix"] == {
        "tn": 1,
        "fp": 0,
        "fn": 0,
        "tp": 1,
    }


def test_label_mapping() -> None:
    assert label_to_int("notflip") == 0
    assert label_to_int("flip") == 1
    with pytest.raises(ValueError):
        label_to_int("other")
