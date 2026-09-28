from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.evaluation import (
    DECISION_THRESHOLD,
    compute_binary_metrics,
    int_to_label,
    label_to_int,
)
from src.features import FEATURE_SETS, FEATURE_SIZE
from src.training import (
    fit_logistic_classifier,
    positive_probabilities,
    standardized_logistic_coefficients,
)


C_GRID = tuple(10.0 ** exponent for exponent in range(-4, 5))
TUNING_SEED = 42
TUNING_VALIDATION_FRACTION = 0.10


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stable_key(sample_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def frozen_tuning_membership(
    rows: Sequence[Mapping[str, str]],
    *,
    validation_fraction: float = TUNING_VALIDATION_FRACTION,
    seed: int = TUNING_SEED,
) -> dict[str, str]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_label: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    membership: dict[str, str] = {}
    for label in sorted(by_label):
        ordered = sorted(
            by_label[label],
            key=lambda row: _stable_key(row["sample_id"], seed=seed),
        )
        validation_n = max(1, int(np.ceil(validation_fraction * len(ordered))))
        validation_ids = {row["sample_id"] for row in ordered[-validation_n:]}
        for row in ordered:
            membership[row["sample_id"]] = (
                "validation" if row["sample_id"] in validation_ids else "fit"
            )
    return membership


def feature_matrix(
    rows: Sequence[Mapping[str, str]], feature_names: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains invalid values")
    return matrix


def prediction_rows(
    rows: Sequence[Mapping[str, str]],
    probabilities: np.ndarray,
    *,
    model_name: str,
) -> list[dict[str, object]]:
    output = []
    for row, probability in zip(rows, probabilities, strict=True):
        truth = label_to_int(row["label"])
        prediction = int(float(probability) >= DECISION_THRESHOLD)
        output.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "true_label": row["label"],
                "true_class": truth,
                "probability_flip": float(probability),
                "predicted_class": prediction,
                "predicted_label": int_to_label(prediction),
                "correct": str(prediction == truth).lower(),
                "model": model_name,
            }
        )
    return output


def _metric_sort_key(record: Mapping[str, object]) -> tuple[float, float]:
    return (-float(record["f1"]), float(record["C"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible MonReader majority and handcrafted-logistic baselines."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("manifests/splits/SPLIT-001_original.csv"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_rows = read_csv(args.features)
    split_rows = read_csv(args.split)
    if len(feature_rows) != 2989 or len(split_rows) != 2989:
        raise ValueError("Expected the complete 2,989-row canonical dataset")

    feature_by_id = {row["sample_id"]: row for row in feature_rows}
    if len(feature_by_id) != len(feature_rows):
        raise ValueError("Feature sample IDs are not unique")
    split_by_id = {row["sample_id"]: row for row in split_rows}
    if set(feature_by_id) != set(split_by_id):
        raise ValueError("Feature and split sample IDs do not match")

    train_rows = [
        feature_by_id[row["sample_id"]]
        for row in split_rows
        if row["partition"] == "train"
    ]
    test_rows = [
        feature_by_id[row["sample_id"]]
        for row in split_rows
        if row["partition"] == "test"
    ]
    if len(train_rows) != 2392 or len(test_rows) != 597:
        raise ValueError("Original split counts changed unexpectedly")

    membership = frozen_tuning_membership(train_rows)
    fit_rows = [
        row for row in train_rows if membership[row["sample_id"]] == "fit"
    ]
    validation_rows = [
        row
        for row in train_rows
        if membership[row["sample_id"]] == "validation"
    ]
    y_fit = np.asarray(
        [label_to_int(row["label"]) for row in fit_rows], dtype=np.int64
    )
    y_validation = np.asarray(
        [label_to_int(row["label"]) for row in validation_rows], dtype=np.int64
    )
    y_full_train = np.asarray(
        [label_to_int(row["label"]) for row in train_rows], dtype=np.int64
    )
    y_test = np.asarray(
        [label_to_int(row["label"]) for row in test_rows], dtype=np.int64
    )

    output_root = args.output_root
    tables_dir = output_root / "outputs" / "tables"
    metrics_dir = output_root / "outputs" / "metrics"
    predictions_dir = output_root / "outputs" / "predictions"
    for directory in (tables_dir, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    membership_rows = [
        {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "partition": membership[row["sample_id"]],
        }
        for row in sorted(train_rows, key=lambda item: item["sample_id"])
    ]
    write_csv(
        tables_dir / "baseline_tuning_membership.csv",
        membership_rows,
        ("sample_id", "label", "partition"),
    )

    fit_prior = float(y_fit.mean())
    full_prior = float(y_full_train.mean())
    majority_validation_probability = np.full(
        len(validation_rows), fit_prior, dtype=np.float64
    )
    majority_test_probability = np.full(
        len(test_rows), full_prior, dtype=np.float64
    )
    majority_validation_metrics = compute_binary_metrics(
        y_validation, majority_validation_probability
    )
    majority_test_metrics = compute_binary_metrics(
        y_test, majority_test_probability
    )

    majority_validation_predictions = prediction_rows(
        validation_rows,
        majority_validation_probability,
        model_name="majority_class",
    )
    majority_test_predictions = prediction_rows(
        test_rows, majority_test_probability, model_name="majority_class"
    )
    prediction_fields = (
        "sample_id",
        "video_id",
        "frame_number",
        "true_label",
        "true_class",
        "probability_flip",
        "predicted_class",
        "predicted_label",
        "correct",
        "model",
    )
    write_csv(
        predictions_dir / "majority_original_validation.csv",
        majority_validation_predictions,
        prediction_fields,
    )
    write_csv(
        predictions_dir / "majority_original_test.csv",
        majority_test_predictions,
        prediction_fields,
    )

    c_search_rows: list[dict[str, object]] = []
    best_per_set: dict[str, dict[str, object]] = {}
    validation_predictions_by_set: dict[
        str, tuple[object, np.ndarray, Sequence[str]]
    ] = {}

    for set_name in ("A", "B", "C"):
        feature_names = FEATURE_SETS[set_name]
        x_fit = feature_matrix(fit_rows, feature_names)
        x_validation = feature_matrix(validation_rows, feature_names)
        candidates: list[dict[str, object]] = []
        for c_value in C_GRID:
            model = fit_logistic_classifier(x_fit, y_fit, c_value=c_value)
            probabilities = positive_probabilities(model, x_validation)
            metrics = compute_binary_metrics(y_validation, probabilities)
            record: dict[str, object] = {
                "feature_set": set_name,
                "feature_count": len(feature_names),
                "C": c_value,
            }
            record.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                }
            )
            record.update(
                {
                    f"cm_{key}": value
                    for key, value in metrics["confusion_matrix"].items()
                }
            )
            candidates.append(record)
            c_search_rows.append(record)
        best = sorted(candidates, key=_metric_sort_key)[0]
        best_per_set[set_name] = best
        best_model = fit_logistic_classifier(
            x_fit, y_fit, c_value=float(best["C"])
        )
        best_probabilities = positive_probabilities(
            best_model, x_validation
        )
        validation_predictions_by_set[set_name] = (
            best_model,
            best_probabilities,
            feature_names,
        )

    selected_set = sorted(
        best_per_set,
        key=lambda set_name: (
            -float(best_per_set[set_name]["f1"]),
            len(FEATURE_SETS[set_name]),
            float(best_per_set[set_name]["C"]),
        ),
    )[0]
    selected_c = float(best_per_set[selected_set]["C"])
    _, selected_validation_probabilities, selected_features = (
        validation_predictions_by_set[selected_set]
    )
    selected_validation_metrics = compute_binary_metrics(
        y_validation, selected_validation_probabilities
    )
    selected_validation_predictions = prediction_rows(
        validation_rows,
        selected_validation_probabilities,
        model_name=f"handcrafted_logistic_set_{selected_set}",
    )
    write_csv(
        predictions_dir / "handcrafted_logistic_original_validation.csv",
        selected_validation_predictions,
        prediction_fields,
    )

    final_model = fit_logistic_classifier(
        feature_matrix(train_rows, selected_features),
        y_full_train,
        c_value=selected_c,
    )
    final_test_probabilities = positive_probabilities(
        final_model, feature_matrix(test_rows, selected_features)
    )
    final_test_metrics = compute_binary_metrics(
        y_test, final_test_probabilities
    )
    final_test_predictions = prediction_rows(
        test_rows,
        final_test_probabilities,
        model_name=f"handcrafted_logistic_set_{selected_set}",
    )
    write_csv(
        predictions_dir / "handcrafted_logistic_original_test.csv",
        final_test_predictions,
        prediction_fields,
    )

    search_fields = (
        "feature_set",
        "feature_count",
        "C",
        "threshold",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "n",
        "positive_count",
        "negative_count",
        "cm_tn",
        "cm_fp",
        "cm_fn",
        "cm_tp",
    )
    write_csv(
        tables_dir / "logistic_c_search.csv",
        c_search_rows,
        search_fields,
    )

    coefficient_rows = [
        {
            "feature_set": selected_set,
            "C": selected_c,
            "feature": feature,
            "standardized_coefficient": coefficient,
        }
        for feature, coefficient in standardized_logistic_coefficients(
            final_model, selected_features
        )
    ]
    coefficient_rows.sort(
        key=lambda row: (
            -abs(float(row["standardized_coefficient"])),
            str(row["feature"]),
        )
    )
    write_csv(
        tables_dir / "handcrafted_logistic_coefficients.csv",
        coefficient_rows,
        (
            "feature_set",
            "C",
            "feature",
            "standardized_coefficient",
        ),
    )

    summary = {
        "source_commit": args.source_commit,
        "dataset_id": "DATA-001",
        "evaluation_split": "Original",
        "evaluation_split_id": "SPLIT-001",
        "positive_class": "flip",
        "threshold": DECISION_THRESHOLD,
        "feature_resolution": {
            "width": FEATURE_SIZE[0],
            "height": FEATURE_SIZE[1],
        },
        "feature_csv": {
            "path": "outputs/tables/DATA-001_handcrafted_features.csv",
            "sha256": sha256_file(args.features),
            "row_count": len(feature_rows),
        },
        "tuning": {
            "seed": TUNING_SEED,
            "validation_fraction": TUNING_VALIDATION_FRACTION,
            "fit_count": len(fit_rows),
            "validation_count": len(validation_rows),
            "C_grid": list(C_GRID),
            "penalty": "l2",
            "standardization": (
                "fit-partition mean/std only during selection; "
                "full supplied-training mean/std for final refit"
            ),
        },
        "majority": {
            "validation_training_positive_prior": fit_prior,
            "test_training_positive_prior": full_prior,
            "validation_metrics": majority_validation_metrics,
            "test_metrics": majority_test_metrics,
        },
        "handcrafted_logistic": {
            "sets": {
                name: list(FEATURE_SETS[name])
                for name in ("A", "B", "C")
            },
            "best_per_set": best_per_set,
            "selected_feature_set": selected_set,
            "selected_C": selected_c,
            "validation_metrics": selected_validation_metrics,
            "test_metrics": final_test_metrics,
        },
    }
    summary_path = metrics_dir / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "majority_test_f1": majority_test_metrics["f1"],
                "selected_feature_set": selected_set,
                "selected_C": selected_c,
                "logistic_validation_f1": selected_validation_metrics["f1"],
                "logistic_test_f1": final_test_metrics["f1"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
