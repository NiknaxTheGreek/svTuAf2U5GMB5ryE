#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2026
POSITIVE_LABEL = 1
FEATURE_SIZE = (96, 96)


def entropy32(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize(
            FEATURE_SIZE, Image.Resampling.BILINEAR
        )
        rgb = np.asarray(img, dtype=np.float32) / 255.0

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad = np.sqrt(
        gx[:-1, :] ** 2
        + gy[:, :-1] ** 2
    )

    d2x = np.abs(np.diff(gray, n=2, axis=1))
    d2y = np.abs(np.diff(gray, n=2, axis=0))

    h, w = gray.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = gray[y0:y1, x0:x1]

    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border = gray[border_mask]

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p10": float(np.quantile(gray, 0.10)),
        "gray_p25": float(np.quantile(gray, 0.25)),
        "gray_p50": float(np.quantile(gray, 0.50)),
        "gray_p75": float(np.quantile(gray, 0.75)),
        "gray_p90": float(np.quantile(gray, 0.90)),
        "dark_ratio": float((gray < 0.25).mean()),
        "bright_ratio": float((gray > 0.75).mean()),
        "entropy_32bin": entropy32(gray),
        "grad_x_mean": float(gx.mean()),
        "grad_y_mean": float(gy.mean()),
        "grad_mean": float(grad.mean()),
        "edge_ratio": float((grad > 0.10).mean()),
        "second_diff_mean": float((d2x.mean() + d2y.mean()) / 2.0),
        "red_mean": float(rgb[:, :, 0].mean()),
        "green_mean": float(rgb[:, :, 1].mean()),
        "blue_mean": float(rgb[:, :, 2].mean()),
        "red_std": float(rgb[:, :, 0].std()),
        "green_std": float(rgb[:, :, 1].std()),
        "blue_std": float(rgb[:, :, 2].std()),
        "center_gray_mean": float(center.mean()),
        "border_gray_mean": float(border.mean()),
        "center_minus_border": float(center.mean() - border.mean()),
        "left_minus_right": float(gray[:, : w // 2].mean() - gray[:, w // 2 :].mean()),
        "top_minus_bottom": float(gray[: h // 2, :].mean() - gray[h // 2 :, :].mean()),
    }


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support": int(len(y_true)),
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if (dataset_root / "images").is_dir():
        dataset_root = dataset_root / "images"

    frames = pd.read_csv(
        args.frames,
        dtype={"video_id_str": str, "source_group_id": str},
    )
    split = pd.read_csv(
        args.split,
        dtype={"video_id": str, "source_group_id": str},
    )

    required_frame_cols = {"path", "source_group_id", "target", "label"}
    required_split_cols = {"source_group_id", "partition"}
    if missing := required_frame_cols - set(frames.columns):
        raise ValueError(f"Frame table missing columns: {sorted(missing)}")
    if missing := required_split_cols - set(split.columns):
        raise ValueError(f"Split table missing columns: {sorted(missing)}")

    merged = frames.merge(
        split[["source_group_id", "partition"]],
        on="source_group_id",
        how="left",
        validate="many_to_one",
    )
    if merged["partition"].isna().any():
        raise AssertionError("At least one frame has no frozen partition")
    if len(merged) != 2989:
        raise AssertionError(f"Expected 2,989 total frames, got {len(merged)}")

    # The frozen test partition is deliberately excluded before any pixels are opened.
    dev = merged.loc[merged["partition"].isin(["train", "validation"])].copy()
    if len(dev) != 2392:
        raise AssertionError(f"Expected 2,392 train+validation frames, got {len(dev)}")

    records: list[dict] = []
    for row in dev.itertuples(index=False):
        img_path = dataset_root / row.path
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        feats = image_features(img_path)
        records.append(
            {
                "path": row.path,
                "source_group_id": row.source_group_id,
                "partition": row.partition,
                "label": row.label,
                "target": int(row.target),
                **feats,
            }
        )

    feature_df = pd.DataFrame(records)
    feature_cols = [
        c
        for c in feature_df.columns
        if c
        not in {
            "path",
            "source_group_id",
            "partition",
            "label",
            "target",
        }
    ]

    train = feature_df.loc[feature_df["partition"] == "train"].copy()
    val = feature_df.loc[feature_df["partition"] == "validation"].copy()
    if len(train) != 1795 or len(val) != 597:
        raise AssertionError(
            f"Unexpected development split sizes: train={len(train)}, val={len(val)}"
        )

    X_train = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["target"].to_numpy(dtype=int)
    X_val = val[feature_cols].to_numpy(dtype=np.float64)
    y_val = val["target"].to_numpy(dtype=int)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    random_state=SEED,
                    max_iter=1000,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]

    # Fixed threshold by design: no validation threshold tuning.
    threshold = 0.5
    train_pred = (train_prob >= threshold).astype(int)
    val_pred = (val_prob >= threshold).astype(int)

    always_notflip = np.zeros_like(y_val)
    always_flip = np.ones_like(y_val)

    metrics = {
        "baseline_definition": {
            "model": "StandardScaler + LogisticRegression",
            "features": "26 simple global image-statistics features from 96x96 RGB resize",
            "C": 1.0,
            "solver": "liblinear",
            "seed": SEED,
            "decision_threshold": threshold,
            "threshold_tuned_on_validation": False,
            "test_pixels_opened": False,
            "test_metrics_computed": False,
        },
        "train": metric_block(y_train, train_pred),
        "validation": metric_block(y_val, val_pred),
        "sanity_references_validation": {
            "always_notflip": metric_block(y_val, always_notflip),
            "always_flip": metric_block(y_val, always_flip),
        },
        "primary_metric": "F1 for positive class flip",
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
    }

    logistic = model.named_steps["logistic"]
    coefficients = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient_standardized": logistic.coef_[0],
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient_standardized"].abs()
    coefficients = coefficients.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    val_predictions = val[
        ["path", "source_group_id", "label", "target"]
    ].copy()
    val_predictions["prob_flip"] = val_prob
    val_predictions["prediction"] = val_pred
    val_predictions["correct"] = val_predictions["target"] == val_predictions["prediction"]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "image_stat_features_train_validation.csv"
    pred_path = out / "validation_predictions.csv"
    coef_path = out / "logistic_coefficients.csv"
    metrics_path = out / "baseline_metrics.json"

    feature_df.to_csv(feature_path, index=False)
    val_predictions.to_csv(pred_path, index=False)
    coefficients.to_csv(coef_path, index=False)

    metrics["artifact_sha256"] = {
        "image_stat_features_train_validation.csv": sha256_file(feature_path),
        "validation_predictions.csv": sha256_file(pred_path),
        "logistic_coefficients.csv": sha256_file(coef_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
