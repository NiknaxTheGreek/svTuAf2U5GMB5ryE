#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_CHECKPOINT_SHA256 = "49f2256715eb293adb2caaac410b6b4ed7dd6a2c3ba99828504f672a244170f1"
EXPECTED_EPOCH = 4
EXPECTED_THRESHOLD = 0.5
IMAGE_SIZE = 160
BATCH_SIZE = 64
NAME_RE = re.compile(r"^(?P<video_id>.+)_(?P<frame_num>\d+)$")


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


def build_test_manifest(images_root: Path, split_csv: Path) -> pd.DataFrame:
    frozen = pd.read_csv(split_csv, dtype={"video_id": str, "source_group_id": str})
    if len(frozen) != 117:
        raise AssertionError(f"Expected 117 frozen source groups, got {len(frozen)}")
    if frozen["source_group_id"].duplicated().any():
        raise AssertionError("Frozen split contains duplicated source groups")

    partition_by_group = dict(zip(frozen["source_group_id"], frozen["partition"]))
    target_by_label = {"notflip": 0, "flip": 1}
    rows = []

    for supplied_split in ("training", "testing"):
        for label in ("flip", "notflip"):
            for path in sorted((images_root / supplied_split / label).glob("*.jpg")):
                m = NAME_RE.match(path.stem)
                if not m:
                    raise ValueError(f"Unexpected filename: {path.name}")
                video_id = f"{int(m.group('video_id')):04d}"
                frame_num = int(m.group("frame_num"))
                source_group_id = f"{label}__{video_id}"
                partition = partition_by_group.get(source_group_id)
                if partition == "test":
                    rows.append(
                        {
                            "path": str(path),
                            "relative_path": path.relative_to(images_root).as_posix(),
                            "label": label,
                            "target": target_by_label[label],
                            "video_id": video_id,
                            "frame_num": frame_num,
                            "source_group_id": source_group_id,
                            "partition": partition,
                        }
                    )

    test_df = pd.DataFrame(rows).sort_values(
        ["target", "source_group_id", "frame_num"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    if len(test_df) != 597:
        raise AssertionError(f"Expected 597 frozen test frames, got {len(test_df)}")
    class_counts = test_df["label"].value_counts().to_dict()
    if class_counts != {"notflip": 307, "flip": 290}:
        raise AssertionError(f"Unexpected test class counts: {class_counts}")
    if test_df["source_group_id"].nunique() != 24:
        raise AssertionError(
            f"Expected 24 frozen test source groups, got {test_df['source_group_id'].nunique()}"
        )
    if test_df.duplicated(["label", "video_id", "frame_num"]).any():
        raise AssertionError("Duplicate canonical source-frame key found in test set")

    return test_df


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["path"]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        y = torch.tensor(int(row["target"]), dtype=torch.long)
        return x, y, index


def metric_block(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = (
        int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(len(y_true)),
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    archive_sha = sha256_file(args.archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise AssertionError(
            f"Archive SHA mismatch: {archive_sha} != {EXPECTED_ARCHIVE_SHA256}"
        )

    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            f"Checkpoint SHA mismatch: {checkpoint_sha} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != EXPECTED_EPOCH:
        raise AssertionError(f"Expected selected epoch {EXPECTED_EPOCH}, got {checkpoint.get('epoch')}")
    if float(checkpoint.get("threshold", -1)) != EXPECTED_THRESHOLD:
        raise AssertionError(
            f"Expected threshold {EXPECTED_THRESHOLD}, got {checkpoint.get('threshold')}"
        )
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise AssertionError(f"Unexpected architecture: {checkpoint.get('architecture')}")
    if int(checkpoint.get("image_size", -1)) != IMAGE_SIZE:
        raise AssertionError(f"Unexpected image size: {checkpoint.get('image_size')}")

    images_root = resolve_images_root(args.dataset_root)
    test_df = build_test_manifest(images_root, args.split)

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = TestDataset(test_df, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs, targets = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x).squeeze(1)
            probs.append(torch.sigmoid(logits).numpy())
            targets.append(y.numpy().astype(int))

    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probs)
    threshold = EXPECTED_THRESHOLD
    metrics = metric_block(y_true, y_prob, threshold)

    predictions = test_df[
        ["relative_path", "source_group_id", "label", "target", "frame_num"]
    ].copy()
    predictions["prob_flip"] = y_prob
    predictions["prediction"] = (y_prob >= threshold).astype(int)
    predictions["correct"] = predictions["target"] == predictions["prediction"]

    errors = predictions.loc[~predictions["correct"]].copy()
    error_by_group = (
        errors.groupby(["source_group_id", "label"])
        .agg(errors=("correct", "size"))
        .reset_index()
        .sort_values(["errors", "source_group_id"], ascending=[False, True])
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "test_predictions.csv"
    err_path = out / "test_errors_by_group.csv"
    manifest_path = out / "test_manifest.csv"
    metrics_path = out / "final_test_metrics.json"

    predictions.to_csv(pred_path, index=False)
    error_by_group.to_csv(err_path, index=False)
    test_df.drop(columns=["path"]).to_csv(manifest_path, index=False)

    result = {
        "status": "FINAL_TEST_COMPLETE",
        "evaluation_type": "one-shot frozen holdout evaluation",
        "architecture": "MobileNetV3-Small",
        "checkpoint_epoch": EXPECTED_EPOCH,
        "checkpoint_sha256": checkpoint_sha,
        "decision_threshold": threshold,
        "threshold_tuned_on_test": False,
        "test": metrics,
        "information_boundary": {
            "test_frames_scored": int(len(test_df)),
            "test_source_groups_scored": int(test_df["source_group_id"].nunique()),
            "test_access_purpose": "final evaluation only",
            "post_test_tuning_allowed": False,
        },
        "artifact_sha256": {
            "test_predictions.csv": sha256_file(pred_path),
            "test_errors_by_group.csv": sha256_file(err_path),
            "test_manifest.csv": sha256_file(manifest_path),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
