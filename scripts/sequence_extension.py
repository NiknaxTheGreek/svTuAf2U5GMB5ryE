#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

EXPECTED_VALIDATION_SHA256 = "e3521f7cf87dd7c1ad8537ab869a452cae62dde7cac785556c3f4a4f1f270d92"
EXPECTED_TEST_SHA256 = "3db3083ddccb0fafe9142f1e183104b3b193311033fcb231c3ba6ca9211822cf"
THRESHOLD = 0.5

# Tie-break order is fixed before test aggregation is evaluated.
CANDIDATE_PRIORITY = [
    "mean_prob",
    "median_prob",
    "positive_fraction",
    "p75_prob",
    "p90_prob",
    "top3_mean",
    "max_prob",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby("source_group_id", sort=True):
        probs = g["prob_flip"].to_numpy(dtype=float)
        rows.append(
            {
                "source_group_id": gid,
                "label": str(g["label"].iloc[0]),
                "target": int(g["target"].iloc[0]),
                "frame_count": int(len(g)),
                "mean_prob": float(probs.mean()),
                "median_prob": float(np.median(probs)),
                "positive_fraction": float((probs >= THRESHOLD).mean()),
                "p75_prob": float(np.quantile(probs, 0.75)),
                "p90_prob": float(np.quantile(probs, 0.90)),
                "top3_mean": float(np.sort(probs)[-min(3, len(probs)):].mean()),
                "max_prob": float(probs.max()),
            }
        )
    return pd.DataFrame(rows)


def metric_block(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_pred = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--test-predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if sha256_file(args.validation_predictions) != EXPECTED_VALIDATION_SHA256:
        raise AssertionError("Validation predictions hash mismatch")
    if sha256_file(args.test_predictions) != EXPECTED_TEST_SHA256:
        raise AssertionError("Test predictions hash mismatch")

    val = pd.read_csv(args.validation_predictions)
    test = pd.read_csv(args.test_predictions)

    required = {"source_group_id", "label", "target", "frame_num", "prob_flip"}
    for name, df in (("validation", val), ("test", test)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")

    if len(val) != 597 or val["source_group_id"].nunique() != 23:
        raise AssertionError("Validation population changed")
    if len(test) != 597 or test["source_group_id"].nunique() != 24:
        raise AssertionError("Test population changed")

    val_seq = aggregate(val)
    test_seq = aggregate(test)

    candidate_rows = []
    for method in CANDIDATE_PRIORITY:
        m = metric_block(
            val_seq["target"].to_numpy(dtype=int),
            val_seq[method].to_numpy(dtype=float),
        )
        candidate_rows.append(
            {
                "method": method,
                "selection_priority": CANDIDATE_PRIORITY.index(method),
                **m,
            }
        )
    candidates = pd.DataFrame(candidate_rows)

    best_f1 = float(candidates["f1"].max())
    tied = candidates.loc[np.isclose(candidates["f1"], best_f1)].sort_values(
        "selection_priority"
    )
    selected_method = str(tied.iloc[0]["method"])
    selected_validation = metric_block(
        val_seq["target"].to_numpy(dtype=int),
        val_seq[selected_method].to_numpy(dtype=float),
    )

    # Test is evaluated only for the validation-selected aggregation rule.
    test_scores = test_seq[selected_method].to_numpy(dtype=float)
    selected_test = metric_block(
        test_seq["target"].to_numpy(dtype=int),
        test_scores,
    )
    test_seq = test_seq.copy()
    test_seq["selected_score"] = test_scores
    test_seq["prediction"] = (test_scores >= THRESHOLD).astype(int)
    test_seq["correct"] = test_seq["prediction"] == test_seq["target"]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    val_path = out / "sequence_validation_candidates.csv"
    test_path = out / "sequence_test_predictions.csv"
    metrics_path = out / "sequence_metrics.json"
    candidates.to_csv(val_path, index=False)
    test_seq[
        [
            "source_group_id",
            "label",
            "target",
            "frame_count",
            "selected_score",
            "prediction",
            "correct",
        ]
    ].to_csv(test_path, index=False)

    result = {
        "status": "SEQUENCE_EXTENSION_COMPLETE",
        "governance": {
            "extension_type": "post-holdout exploratory downstream analysis",
            "frame_model_changed": False,
            "frame_threshold_changed": False,
            "aggregation_selected_using": "validation only",
            "test_used_for_aggregation_selection": False,
            "reported_single_frame_final_test_result_changed": False,
        },
        "candidate_methods": CANDIDATE_PRIORITY,
        "sequence_decision_threshold": THRESHOLD,
        "selection_tie_break": "highest validation F1, then fixed candidate priority",
        "selected_method": selected_method,
        "validation_sequence": selected_validation,
        "test_sequence_exploratory": selected_test,
        "validation_source_groups": int(len(val_seq)),
        "test_source_groups": int(len(test_seq)),
        "artifact_sha256": {
            "sequence_validation_candidates.csv": sha256_file(val_path),
            "sequence_test_predictions.csv": sha256_file(test_path),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
