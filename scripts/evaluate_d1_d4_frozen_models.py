#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_SIZE = 160
BATCH_SIZE = 64
THRESHOLD = 0.5
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_images_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "training").is_dir() and (root / "testing").is_dir():
        return root
    if (root / "images" / "training").is_dir() and (root / "images" / "testing").is_dir():
        return root / "images"
    raise FileNotFoundError(f"Could not resolve image root from {root}")


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: Path, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(self.images_root / row["path"]) as im:
            x = self.transform(im.convert("RGB"))
        y = torch.tensor(float(row["target"]), dtype=torch.float32)
        return x, y, idx


def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x).squeeze(1)
            targets.append(y.numpy().astype(int))
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(targets), np.concatenate(probs)


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(len(y_true)),
        "flip_support": int((y_true == 1).sum()),
        "notflip_support": int((y_true == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--model-manifest", required=True, type=Path)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--d2-manifest", required=True, type=Path)
    ap.add_argument("--d3-manifest", required=True, type=Path)
    ap.add_argument("--d4-manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("Authoritative archive SHA mismatch")

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if model_manifest["status"] != "FROZEN":
        raise SystemExit("Final models are not frozen")
    if model_manifest["protocol_id"] != protocol["protocol_id"]:
        raise SystemExit("Final model/protocol mismatch")
    if model_manifest["threshold"] != THRESHOLD:
        raise SystemExit("Frozen threshold changed")

    expected_hashes = {
        x["dataset_id"]: x["checkpoint_sha256"] for x in model_manifest["models"]
    }
    selected_epoch = int(model_manifest["selected_epoch"])

    base = pd.read_csv(args.d2_manifest, dtype={"video_id": str})
    d3 = pd.read_csv(args.d3_manifest, dtype={"video_id": str})
    d4 = pd.read_csv(args.d4_manifest, dtype={"video_id": str})
    target_map = {"notflip": 0, "flip": 1}

    tests = {
        "D1": base.loc[base["supplied_split"] == "testing"].copy(),
        "D2": base.loc[base["d2_split"] == "test"].copy(),
        "D3": d3.loc[d3["d3_split"] == "testing"].copy(),
        "D4": d4.loc[d4["d4_split"] == "test"].copy(),
    }
    expected_counts = {"D1": 597, "D2": 597, "D3": 391, "D4": 408}
    if {k: len(v) for k, v in tests.items()} != expected_counts:
        raise SystemExit("Frozen test populations changed")

    for df in tests.values():
        df["target"] = df["label"].map(target_map)
        if df["target"].isna().any():
            raise SystemExit("Unknown test label")

    # Re-assert the deployment-relevant isolation invariant immediately before
    # consuming D2/D4 test pixels.
    for did, df, split_field, train_value in [
        ("D2", base, "d2_split", "train"),
        ("D4", d4, "d4_split", "train"),
    ]:
        test_value = "test"
        train_df = df.loc[df[split_field] == train_value]
        test_df = df.loc[df[split_field] == test_value]
        if set(train_df["environment_group_id"]) & set(test_df["environment_group_id"]):
            raise SystemExit(f"{did}: environment overlap before final evaluation")
        if set(train_df["sequence_id"]) & set(test_df["sequence_id"]):
            raise SystemExit(f"{did}: sequence overlap before final evaluation")
        if set(train_df["path"]) & set(test_df["path"]):
            raise SystemExit(f"{did}: frame overlap before final evaluation")

    images_root = resolve_images_root(args.dataset_root)
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    prediction_hashes = {}

    for did in ("D1", "D2", "D3", "D4"):
        checkpoint_path = args.models_dir / f"{did.lower()}_mobilenetv3_small.pt"
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hashes[did]:
            raise SystemExit(
                f"{did}: checkpoint hash mismatch {actual_hash} != {expected_hashes[did]}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["dataset_id"] != did:
            raise SystemExit(f"{did}: checkpoint dataset ID mismatch")
        if int(checkpoint["selected_epoch"]) != selected_epoch:
            raise SystemExit(f"{did}: checkpoint epoch mismatch")
        if float(checkpoint["threshold"]) != THRESHOLD:
            raise SystemExit(f"{did}: checkpoint threshold mismatch")

        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        ds = EvalDataset(tests[did], images_root, eval_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        y_true, y_prob = evaluate_model(model, loader)
        m = metrics(y_true, y_prob)
        results[did] = m

        pred = tests[did][[
            "path", "sequence_id", "environment_group_id", "label", "frame_num"
        ]].copy().reset_index(drop=True)
        pred["target"] = y_true
        pred["prob_flip"] = y_prob
        pred["prediction"] = (y_prob >= THRESHOLD).astype(int)
        pred["correct"] = pred["target"] == pred["prediction"]
        pred_path = out / f"{did.lower()}_test_predictions.csv"
        pred.to_csv(pred_path, index=False)
        prediction_hashes[pred_path.name] = sha256_file(pred_path)

    comparison = pd.DataFrame([
        {
            "dataset_id": did,
            "test_frames": results[did]["support"],
            "flip_support": results[did]["flip_support"],
            "notflip_support": results[did]["notflip_support"],
            "f1": results[did]["f1"],
            "precision": results[did]["precision"],
            "recall": results[did]["recall"],
            "accuracy": results[did]["accuracy"],
            "balanced_accuracy": results[did]["balanced_accuracy"],
            "tn": results[did]["tn"],
            "fp": results[did]["fp"],
            "fn": results[did]["fn"],
            "tp": results[did]["tp"],
        }
        for did in ("D1", "D2", "D3", "D4")
    ])
    comparison_path = out / "d1_d4_test_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "status": "COMPLETE",
        "evaluation_type": "one_locked_run_after_all_four_model_hashes_frozen",
        "protocol_id": protocol["protocol_id"],
        "selected_epoch": selected_epoch,
        "threshold": THRESHOLD,
        "D2_holdout_status": "CONSUMED",
        "D4_holdout_status": "CONSUMED",
        "metrics": results,
        "descriptive_f1_differences": {
            "D2_minus_D1": results["D2"]["f1"] - results["D1"]["f1"],
            "D3_minus_D1": results["D3"]["f1"] - results["D1"]["f1"],
            "D4_minus_D2": results["D4"]["f1"] - results["D2"]["f1"],
            "D4_minus_D3": results["D4"]["f1"] - results["D3"]["f1"],
        },
        "interpretation_guardrail": (
            "D1-D4 use different test populations by design. F1 differences are "
            "descriptive experiment contrasts, not causal estimates of the isolated "
            "effect of source grouping or deduplication."
        ),
        "claim_scope": protocol["final_evaluation_rule"]["claim_scope"],
        "post_evaluation_rule": (
            "No tuning or redesign is permitted using D2/D4 test results. Any new "
            "model or methodological change requires a new protected evaluation design."
        ),
        "artifact_sha256": {
            **prediction_hashes,
            comparison_path.name: sha256_file(comparison_path),
        },
    }
    summary_path = out / "final_d1_d4_evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "status": "PASS",
        "checkpoint_hashes_verified": True,
        "D2_environment_overlap_pre_eval": 0,
        "D2_sequence_overlap_pre_eval": 0,
        "D4_environment_overlap_pre_eval": 0,
        "D4_sequence_overlap_pre_eval": 0,
        "all_four_evaluated_in_same_invocation": True,
        "final_evaluation_sha256": sha256_file(summary_path),
        "comparison_sha256": sha256_file(comparison_path),
    }
    receipt_path = out / "final_evaluation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
