#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

EXPECTED_CHECKPOINT_SHA256 = "49f2256715eb293adb2caaac410b6b4ed7dd6a2c3ba99828504f672a244170f1"
EXPECTED_PREDICTIONS_SHA256 = "3db3083ddccb0fafe9142f1e183104b3b193311033fcb231c3ba6ca9211822cf"
IMAGE_SIZE = 160
THRESHOLD = 0.5
SEED = 2026
NAME_RE = re.compile(r"^(?P<video_id>.+)_(?P<frame_num>\d+)$")

PERTURBATIONS = {
    "brightness_0.85": ("brightness", 0.85),
    "brightness_1.15": ("brightness", 1.15),
    "contrast_0.85": ("contrast", 0.85),
    "contrast_1.15": ("contrast", 1.15),
    "gaussian_blur_r1": ("blur", 1.0),
    "jpeg_quality_70": ("jpeg", 70),
}


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
    raise FileNotFoundError(root)


def build_path_map(images_root: Path) -> dict[str, Path]:
    mapping = {}
    for supplied_split in ("training", "testing"):
        for label in ("flip", "notflip"):
            for path in (images_root / supplied_split / label).glob("*.jpg"):
                rel = path.relative_to(images_root).as_posix()
                mapping[rel] = path
    return mapping


def counts_to_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return counts_to_metrics(tn, fp, fn, tp)


def source_group_bootstrap(pred: pd.DataFrame, n_boot: int = 20000) -> dict:
    group_rows = []
    for gid, g in pred.groupby("source_group_id"):
        y = g["target"].to_numpy(dtype=int)
        p = g["prediction"].to_numpy(dtype=int)
        m = metrics_from_arrays(y, p)
        group_rows.append(
            {
                "source_group_id": gid,
                "label": g["label"].iloc[0],
                **{k: m[k] for k in ("tn", "fp", "fn", "tp")},
            }
        )
    groups = pd.DataFrame(group_rows)
    flip = groups.loc[groups["label"] == "flip"].reset_index(drop=True)
    notflip = groups.loc[groups["label"] == "notflip"].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    draws = np.empty((n_boot, 4), dtype=float)
    for i in range(n_boot):
        sampled_flip = flip.iloc[rng.integers(0, len(flip), size=len(flip))]
        sampled_notflip = notflip.iloc[rng.integers(0, len(notflip), size=len(notflip))]
        b = pd.concat([sampled_flip, sampled_notflip], ignore_index=True)
        tn, fp, fn, tp = (int(b[k].sum()) for k in ("tn", "fp", "fn", "tp"))
        m = counts_to_metrics(tn, fp, fn, tp)
        draws[i] = [m["f1"], m["precision"], m["recall"], m["accuracy"]]

    out = {}
    for j, metric in enumerate(("f1", "precision", "recall", "accuracy")):
        q = np.quantile(draws[:, j], [0.025, 0.5, 0.975])
        out[metric] = {
            "lower_95": float(q[0]),
            "median": float(q[1]),
            "upper_95": float(q[2]),
        }
    return {
        "method": "stratified source-group bootstrap",
        "seed": SEED,
        "replicates": n_boot,
        "flip_groups_resampled_per_replicate": int(len(flip)),
        "notflip_groups_resampled_per_replicate": int(len(notflip)),
        "intervals": out,
    }


def confidence_analysis(pred: pd.DataFrame) -> dict:
    work = pred.copy()
    work["margin"] = (work["prob_flip"] - 0.5).abs()
    bins = [-1e-12, 0.05, 0.10, 0.20, 0.30, 0.40, 0.5000001]
    labels = ["0-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50"]
    work["margin_bin"] = pd.cut(work["margin"], bins=bins, labels=labels)
    rows = []
    for name, g in work.groupby("margin_bin", observed=True):
        rows.append(
            {
                "margin_bin": str(name),
                "frames": int(len(g)),
                "errors": int((~g["correct"]).sum()),
                "error_rate": float((~g["correct"]).mean()),
            }
        )
    return {
        "mean_margin_correct": float(work.loc[work["correct"], "margin"].mean()),
        "mean_margin_error": float(work.loc[~work["correct"], "margin"].mean()),
        "bins": rows,
    }


def sequence_analysis(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for gid, g in pred.sort_values(["source_group_id", "frame_num"]).groupby("source_group_id"):
        p = g["prediction"].to_numpy(dtype=int)
        err = (~g["correct"]).to_numpy(dtype=bool)
        transitions = int((p[1:] != p[:-1]).sum()) if len(p) > 1 else 0
        longest = current = 0
        for value in err:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        rows.append(
            {
                "source_group_id": gid,
                "label": g["label"].iloc[0],
                "frames": int(len(g)),
                "errors": int(err.sum()),
                "error_rate": float(err.mean()),
                "prediction_transitions": transitions,
                "longest_consecutive_error_run": int(longest),
                "mean_prob_flip": float(g["prob_flip"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["errors", "longest_consecutive_error_run", "source_group_id"],
        ascending=[False, False, True],
    )
    return {
        "groups": int(len(table)),
        "groups_with_zero_errors": int((table["errors"] == 0).sum()),
        "groups_with_any_error": int((table["errors"] > 0).sum()),
        "median_prediction_transitions_per_group": float(table["prediction_transitions"].median()),
        "max_prediction_transitions_in_group": int(table["prediction_transitions"].max()),
        "max_consecutive_error_run": int(table["longest_consecutive_error_run"].max()),
    }, table


def position_analysis(pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for gid, g in pred.groupby("source_group_id"):
        g = g.sort_values("frame_num").copy()
        n = len(g)
        g["relative_position"] = np.linspace(0.0, 1.0, n) if n > 1 else 0.5
        parts.append(g)
    work = pd.concat(parts, ignore_index=True)
    work["position_quintile"] = pd.cut(
        work["relative_position"],
        [-1e-12, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    rows = []
    for label in ("all", "flip", "notflip"):
        sub = work if label == "all" else work.loc[work["label"] == label]
        for q, g in sub.groupby("position_quintile", observed=True):
            rows.append(
                {
                    "label_scope": label,
                    "position_quintile": str(q),
                    "frames": int(len(g)),
                    "errors": int((~g["correct"]).sum()),
                    "error_rate": float((~g["correct"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def perturb_image(img: Image.Image, kind: str, value: float) -> Image.Image:
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(value)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(value)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(value))
        buf.seek(0)
        with Image.open(buf) as decoded:
            return decoded.convert("RGB").copy()
    raise ValueError(kind)


class PerturbedDataset(Dataset):
    def __init__(self, pred: pd.DataFrame, path_map: dict[str, Path], transform, perturbation):
        self.pred = pred.reset_index(drop=True)
        self.path_map = path_map
        self.transform = transform
        self.perturbation = perturbation

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):
        row = self.pred.iloc[idx]
        path = self.path_map[row["relative_path"]]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.perturbation is not None:
                img = perturb_image(img, *self.perturbation)
            x = self.transform(img)
        return x, int(row["target"])


def score_perturbation(model, dataset) -> dict:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x).squeeze(1)
            probs.append(torch.sigmoid(logits).numpy())
            targets.append(y.numpy().astype(int))
    p = np.concatenate(probs)
    y = np.concatenate(targets)
    pred = (p >= THRESHOLD).astype(int)
    return metrics_from_arrays(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--final-predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("Frozen checkpoint hash mismatch")
    if sha256_file(args.final_predictions) != EXPECTED_PREDICTIONS_SHA256:
        raise AssertionError("Final predictions hash mismatch")

    pred = pd.read_csv(args.final_predictions)
    if len(pred) != 597 or pred["source_group_id"].nunique() != 24:
        raise AssertionError("Final holdout prediction population changed")
    pred["correct"] = pred["correct"].astype(bool)

    original = metrics_from_arrays(
        pred["target"].to_numpy(dtype=int),
        pred["prediction"].to_numpy(dtype=int),
    )

    bootstrap = source_group_bootstrap(pred)
    confidence = confidence_analysis(pred)
    seq_summary, seq_table = sequence_analysis(pred)
    pos_table = position_analysis(pred)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
    images_root = resolve_images_root(args.dataset_root)
    path_map = build_path_map(images_root)
    missing = sorted(set(pred["relative_path"]) - set(path_map))
    if missing:
        raise AssertionError(f"Missing holdout files: {missing[:10]}")

    robustness_rows = []
    for name, spec in PERTURBATIONS.items():
        ds = PerturbedDataset(pred, path_map, eval_transform, spec)
        m = score_perturbation(model, ds)
        robustness_rows.append(
            {
                "condition": name,
                **m,
                "f1_delta_vs_original": float(m["f1"] - original["f1"]),
                "accuracy_delta_vs_original": float(m["accuracy"] - original["accuracy"]),
            }
        )
        print(json.dumps(robustness_rows[-1]))

    robustness = pd.DataFrame(robustness_rows).sort_values("f1_delta_vs_original")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    seq_path = out / "source_group_diagnostics.csv"
    pos_path = out / "frame_position_diagnostics.csv"
    rob_path = out / "perturbation_robustness.csv"
    summary_path = out / "post_holdout_analysis.json"
    seq_table.to_csv(seq_path, index=False)
    pos_table.to_csv(pos_path, index=False)
    robustness.to_csv(rob_path, index=False)

    summary = {
        "status": "POST_HOLDOUT_DIAGNOSTIC_COMPLETE",
        "governance": {
            "reported_final_test_metrics_changed": False,
            "model_changed": False,
            "threshold_changed": False,
            "post_test_tuning_performed": False,
            "analysis_only": True,
        },
        "original_final_test_metrics": original,
        "source_group_bootstrap": bootstrap,
        "confidence_analysis": confidence,
        "sequence_analysis": seq_summary,
        "robustness_conditions": robustness.to_dict(orient="records"),
        "artifact_sha256": {
            "source_group_diagnostics.csv": sha256_file(seq_path),
            "frame_position_diagnostics.csv": sha256_file(pos_path),
            "perturbation_robustness.csv": sha256_file(rob_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
