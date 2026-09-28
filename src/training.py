from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_optimizer(
    parameters: Iterable,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    import torch

    params = list(parameters)
    key = name.lower()
    common = {
        "params": params,
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
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
    coefficients = np.asarray(
        model.named_steps["logistic"].coef_, dtype=float
    )
    if coefficients.shape != (1, len(feature_names)):
        raise ValueError(
            "Coefficient shape does not match feature names"
        )
    return [
        (name, float(value))
        for name, value in zip(
            feature_names, coefficients[0], strict=True
        )
    ]



class EarlyStopping:
    def __init__(self, *, patience: int = 8, max_epochs: int = 50) -> None:
        if patience < 1 or max_epochs < 1:
            raise ValueError("patience and max_epochs must be positive")
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_metric = float("-inf")
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def step(self, metric: float, epoch: int) -> tuple[bool, bool]:
        if not np.isfinite(metric):
            raise ValueError("Early-stopping metric must be finite")
        improved = metric > self.best_metric
        if improved:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience or epoch >= self.max_epochs
        return improved, should_stop


def _binary_training_metrics(labels, probabilities):
    from src.evaluation import compute_binary_metrics

    return compute_binary_metrics(labels, probabilities)


def train_one_epoch(
    model,
    loader,
    optimizer,
    *,
    device,
    amp_enabled: bool,
):
    import torch

    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    count = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
        probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
        labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
        probabilities_all.extend(probabilities.tolist())
    metrics = _binary_training_metrics(labels_all, probabilities_all)
    return {
        "loss": total_loss / count,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


def evaluate_binary_model(model, loader, *, device):
    import torch

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    count = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    sample_ids: list[str] = []
    video_ids: list[str] = []
    frame_numbers: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            count += batch_size
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels_all.extend(labels.cpu().numpy().astype(int).tolist())
            probabilities_all.extend(probabilities.tolist())
            sample_ids.extend(list(batch["sample_id"]))
            video_ids.extend(list(batch["video_id"]))
            frame_numbers.extend([int(value) for value in batch["frame_number"]])
    metrics = _binary_training_metrics(labels_all, probabilities_all)
    metrics["loss"] = total_loss / count
    return {
        "metrics": metrics,
        "labels": labels_all,
        "probabilities": probabilities_all,
        "sample_ids": sample_ids,
        "video_ids": video_ids,
        "frame_numbers": frame_numbers,
    }


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch: int,
    best_validation_f1: float,
    config: dict,
) -> str:
    import hashlib
    from pathlib import Path

    import torch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_validation_f1": float(best_validation_f1),
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return digest


def run_binary_training(
    model,
    train_loader,
    validation_loader,
    optimizer,
    *,
    device,
    checkpoint_path,
    config: dict,
    max_epochs: int = 50,
    patience: int = 8,
):
    amp_enabled = getattr(device, "type", str(device)) == "cuda"
    stopper = EarlyStopping(patience=patience, max_epochs=max_epochs)
    history: list[dict[str, object]] = []
    checkpoint_sha256: str | None = None
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
        )
        validation = evaluate_binary_model(model, validation_loader, device=device)
        validation_metrics = validation["metrics"]
        improved, should_stop = stopper.step(validation_metrics["f1"], epoch)
        if improved:
            checkpoint_sha256 = save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_f1=validation_metrics["f1"],
                config=config,
            )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} "
            f"validation_f1={validation_metrics['f1']:.6f} "
            f"improved={improved}",
            flush=True,
        )
        if should_stop:
            break
    if stopper.best_epoch is None or checkpoint_sha256 is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    return {
        "best_epoch": stopper.best_epoch,
        "best_validation_f1": stopper.best_metric,
        "checkpoint_sha256": checkpoint_sha256,
        "history": history,
    }
