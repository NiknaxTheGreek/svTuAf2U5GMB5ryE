from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


DECISION_THRESHOLD = 0.5
POSITIVE_LABEL = "flip"


def label_to_int(label: str) -> int:
    if label == "flip":
        return 1
    if label == "notflip":
        return 0
    raise ValueError(f"Unexpected label: {label}")


def int_to_label(value: int) -> str:
    if value == 1:
        return "flip"
    if value == 0:
        return "notflip"
    raise ValueError(f"Unexpected binary class: {value}")


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("y_true and probabilities must be same-length 1D arrays")
    if y.size == 0:
        raise ValueError("Cannot evaluate an empty population")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must be binary 0/1")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values in [0,1]")
    if len(np.unique(y)) < 2:
        raise ValueError("ROC-AUC and PR-AUC require both classes")

    predicted = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y, p)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(y.size),
        "positive_count": int(y.sum()),
        "negative_count": int(y.size - y.sum()),
    }
