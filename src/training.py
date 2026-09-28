from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    params = list(parameters)
    key = name.lower()
    common = {"params": params, "lr": learning_rate, "weight_decay": weight_decay}
    if key == "adam":
        return torch.optim.Adam(**common)
    if key == "adamw":
        return torch.optim.AdamW(**common)
    if key == "sgd":
        return torch.optim.SGD(**common, momentum=momentum)
    if key == "rmsprop":
        return torch.optim.RMSprop(**common, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def fit_logistic_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
) -> Pipeline:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Invalid logistic-regression training shapes")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Logistic-regression training data is empty")
    if not np.isfinite(x).all():
        raise ValueError("Logistic-regression features contain NaN or inf")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Logistic regression requires both binary classes")
    if c_value <= 0:
        raise ValueError("C must be positive")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    l1_ratio=0.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x, y)
    return pipeline


def positive_probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Prediction features must be a finite 2D array")
    classes = list(model.named_steps["logistic"].classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected logistic class order: {classes}")
    return model.predict_proba(values)[:, 1]


def standardized_logistic_coefficients(
    model: Pipeline,
    feature_names: Sequence[str],
) -> list[tuple[str, float]]:
    coefficients = np.asarray(model.named_steps["logistic"].coef_, dtype=float)
    if coefficients.shape != (1, len(feature_names)):
        raise ValueError("Coefficient shape does not match feature names")
    return [
        (name, float(value))
        for name, value in zip(feature_names, coefficients[0], strict=True)
    ]
