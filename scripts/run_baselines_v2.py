from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import PIL
import sklearn
import skimage
from PIL import Image
from skimage import color, feature, filters
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

EXPECTED_ARCHIVE_SIZE = 939_921_132
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_MANIFEST_SHA256 = "3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b"
EXPECTED_SPLIT_SHA256 = {
    "O": "d48e182da1b170365b39693317e1edf6dd3395c02ccabdd6a771d7be5e226680",
    "S": "a54cce72ad1f275096f06b98bbc9ec32393e9f112cc0536abd132571e1919619",
    "T": "0a1f39ebb01f14989aad37b87c39fbb874307f81bdc4770d15e8df7c7b9aa625",
    "ST": "ed62ad946187ba33154e84e6732fdcf6110e8309c402b568259bcf2c0e177a85",
}
EXPECTED_ROLE_COUNTS = {
    "O": {"train": 2392, "test": 597},
    "S": {"train": 2219, "test": 770},
    "T": {"train": 2365, "test": 624},
    "ST": {"train": 1756, "test": 161, "context": 1072},
}
CANVAS_W = 224
CANVAS_H = 398
POSITIVE_LABEL = "flip"

A_NAMES = ["brightness", "contrast", "sharpness_laplacian_var", "entropy_64bin", "edge_density_canny_sigma1"]
B_EXTRA_NAMES = [
    "r_mean", "g_mean", "b_mean",
    "r_std", "g_std", "b_std",
    "hue_mean", "hue_std", "saturation_mean", "saturation_std",
    "gray_median", "gray_p10", "gray_p25", "gray_p75", "gray_p90",
]
C_EXTRA_NAMES = (
    [f"gray_hist16_{i:02d}" for i in range(16)]
    + [f"lbp_uniform_p8_r1_{i:02d}" for i in range(10)]
    + [f"grad_orientation9_{i:02d}" for i in range(9)]
)
FEATURE_NAMES = {
    "A": A_NAMES,
    "B": A_NAMES + B_EXTRA_NAMES,
    "C": A_NAMES + B_EXTRA_NAMES + C_EXTRA_NAMES,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resize_pad_rgb(image: Image.Image) -> np.ndarray:
    im = image.convert("RGB")
    scale = min(CANVAS_W / im.width, CANVAS_H / im.height)
    new_w = max(1, min(CANVAS_W, int(round(im.width * scale))))
    new_h = max(1, min(CANVAS_H, int(round(im.height * scale))))
    resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    x = (CANVAS_W - new_w) // 2
    y = (CANVAS_H - new_h) // 2
    canvas.paste(resized, (x, y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def entropy64(gray: np.ndarray) -> float:
    counts, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
    p = counts.astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def norm_hist(values: np.ndarray, bins) -> np.ndarray:
    h, _ = np.histogram(values, bins=bins)
    h = h.astype(np.float64)
    s = h.sum()
    return h / s if s else h


def extract_features(rgb: np.ndarray) -> dict[str, float]:
    gray = color.rgb2gray(rgb).astype(np.float32)
    gray_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)

    lap = filters.laplace(gray, ksize=3)
    edges = feature.canny(gray, sigma=1.0)

    values: dict[str, float] = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness_laplacian_var": float(np.var(lap)),
        "entropy_64bin": entropy64(gray),
        "edge_density_canny_sigma1": float(edges.mean()),
    }

    means = rgb.mean(axis=(0, 1))
    stds = rgb.std(axis=(0, 1))
    values.update({
        "r_mean": float(means[0]),
        "g_mean": float(means[1]),
        "b_mean": float(means[2]),
        "r_std": float(stds[0]),
        "g_std": float(stds[1]),
        "b_std": float(stds[2]),
    })

    hsv = color.rgb2hsv(rgb)
    values.update({
        "hue_mean": float(hsv[..., 0].mean()),
        "hue_std": float(hsv[..., 0].std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "saturation_std": float(hsv[..., 1].std()),
        "gray_median": float(np.median(gray)),
        "gray_p10": float(np.percentile(gray, 10)),
        "gray_p25": float(np.percentile(gray, 25)),
        "gray_p75": float(np.percentile(gray, 75)),
        "gray_p90": float(np.percentile(gray, 90)),
    })

    gh = norm_hist(gray, np.linspace(0.0, 1.0, 17))
    for i, v in enumerate(gh):
        values[f"gray_hist16_{i:02d}"] = float(v)

    lbp = feature.local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    lh = norm_hist(lbp, np.arange(0, 11))
    for i, v in enumerate(lh):
        values[f"lbp_uniform_p8_r1_{i:02d}"] = float(v)

    gx = filters.sobel_h(gray)
    gy = filters.sobel_v(gray)
    mag = np.hypot(gx, gy)
    theta = (np.arctan2(gy, gx) + np.pi) % np.pi
    oh, _ = np.histogram(theta, bins=np.linspace(0.0, np.pi, 10), weights=mag)
    oh = oh.astype(np.float64)
    if oh.sum() > 0:
        oh /= oh.sum()
    for i, v in enumerate(oh):
        values[f"grad_orientation9_{i:02d}"] = float(v)

    missing = set(FEATURE_NAMES["C"]) - set(values)
    if missing:
        raise RuntimeError(f"Missing extracted features: {sorted(missing)}")
    return values


def metric_row(y_true, y_pred, prob=None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": "",
        "pr_auc": "",
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if prob is not None:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
        row["pr_auc"] = float(average_precision_score(y_true, prob))
    return row


def write_dict_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    archive = Path(args.archive)
    manifest_path = Path(args.manifest)
    split_dir = Path(args.split_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    archive_size = archive.stat().st_size
    archive_sha = sha256_file(archive)
    if archive_size != EXPECTED_ARCHIVE_SIZE or archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"Archive identity failure: bytes={archive_size}, sha256={archive_sha}")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Master manifest hash mismatch")

    split_rows = {}
    for regime, expected_hash in EXPECTED_SPLIT_SHA256.items():
        p = split_dir / f"{regime}.csv"
        actual = sha256_file(p)
        if actual != expected_hash:
            raise RuntimeError(f"{regime} split hash mismatch: {actual}")
        sr = read_csv(p)
        rc = Counter(r["role"] for r in sr)
        if dict(rc) != EXPECTED_ROLE_COUNTS[regime]:
            raise RuntimeError(f"{regime} role counts mismatch: {dict(rc)}")
        split_rows[regime] = sr

    manifest = read_csv(manifest_path)
    if len(manifest) != 2989:
        raise RuntimeError(f"Expected 2989 manifest rows, found {len(manifest)}")
    manifest_by_id = {r["sample_id"]: r for r in manifest}
    if len(manifest_by_id) != len(manifest):
        raise RuntimeError("Duplicate sample IDs in master manifest")

    feature_map: dict[str, dict[str, float]] = {}
    with ZipFile(archive) as zf:
        for idx, r in enumerate(sorted(manifest, key=lambda x: x["sample_id"]), 1):
            encoded = zf.read(r["archive_member"])
            with Image.open(BytesIO(encoded)) as im:
                rgb = resize_pad_rgb(im)
            feature_map[r["sample_id"]] = extract_features(rgb)
            if idx % 250 == 0 or idx == len(manifest):
                print(f"feature_extraction_progress={idx}/{len(manifest)}", flush=True)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    scaler_rows: list[dict[str, object]] = []

    for regime in ("O", "S", "T", "ST"):
        members = split_rows[regime]
        train = [r for r in members if r["role"] == "train"]
        test = [r for r in members if r["role"] == "test"]

        y_train = np.array([1 if r["label"] == POSITIVE_LABEL else 0 for r in train], dtype=np.int64)
        y_test = np.array([1 if r["label"] == POSITIVE_LABEL else 0 for r in test], dtype=np.int64)

        train_counts = Counter(r["label"] for r in train)
        if train_counts["flip"] > train_counts["notflip"]:
            majority = "flip"
        else:
            majority = "notflip"
        majority_pred = np.full(len(test), 1 if majority == "flip" else 0, dtype=np.int64)
        m = metric_row(y_test, majority_pred, prob=None)
        metrics_rows.append({
            "regime": regime,
            "baseline": "majority",
            "feature_set": "",
            "train_n": len(train),
            "test_n": len(test),
            "train_flip": int((y_train == 1).sum()),
            "train_notflip": int((y_train == 0).sum()),
            "test_flip": int((y_test == 1).sum()),
            "test_notflip": int((y_test == 0).sum()),
            "majority_label": majority,
            **m,
        })
        for r, yt, yp in zip(test, y_test, majority_pred):
            prediction_rows.append({
                "regime": regime, "baseline": "majority", "feature_set": "",
                "sample_id": r["sample_id"], "video_id": r["video_id"],
                "frame_number": r["frame_number"], "y_true": int(yt),
                "prob_flip": "", "y_pred": int(yp),
            })

        for feature_set in ("A", "B", "C"):
            names = FEATURE_NAMES[feature_set]
            X_train = np.array([[feature_map[r["sample_id"]][n] for n in names] for r in train], dtype=np.float64)
            X_test = np.array([[feature_map[r["sample_id"]][n] for n in names] for r in test], dtype=np.float64)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                class_weight=None,
                max_iter=2000,
                random_state=42,
            )
            model.fit(X_train_s, y_train)
            prob = model.predict_proba(X_test_s)[:, 1]
            pred = (prob >= 0.5).astype(np.int64)

            m = metric_row(y_test, pred, prob=prob)
            metrics_rows.append({
                "regime": regime,
                "baseline": "logistic_regression",
                "feature_set": feature_set,
                "train_n": len(train),
                "test_n": len(test),
                "train_flip": int((y_train == 1).sum()),
                "train_notflip": int((y_train == 0).sum()),
                "test_flip": int((y_test == 1).sum()),
                "test_notflip": int((y_test == 0).sum()),
                "majority_label": "",
                **m,
            })

            for r, yt, p, yp in zip(test, y_test, prob, pred):
                prediction_rows.append({
                    "regime": regime, "baseline": "logistic_regression", "feature_set": feature_set,
                    "sample_id": r["sample_id"], "video_id": r["video_id"],
                    "frame_number": r["frame_number"], "y_true": int(yt),
                    "prob_flip": float(p), "y_pred": int(yp),
                })

            for name, coef, mean, scale in zip(names, model.coef_[0], scaler.mean_, scaler.scale_):
                coefficient_rows.append({
                    "regime": regime, "feature_set": feature_set, "feature": name,
                    "coefficient_standardized": float(coef),
                    "scaler_train_mean": float(mean),
                    "scaler_train_scale": float(scale),
                })
                scaler_rows.append({
                    "regime": regime, "feature_set": feature_set, "feature": name,
                    "train_mean": float(mean), "train_scale": float(scale),
                })
            coefficient_rows.append({
                "regime": regime, "feature_set": feature_set, "feature": "__INTERCEPT__",
                "coefficient_standardized": float(model.intercept_[0]),
                "scaler_train_mean": "", "scaler_train_scale": "",
            })

    metric_fields = [
        "regime","baseline","feature_set","train_n","test_n",
        "train_flip","train_notflip","test_flip","test_notflip","majority_label",
        "f1","precision","recall","accuracy","balanced_accuracy","roc_auc","pr_auc",
        "tn","fp","fn","tp"
    ]
    pred_fields = ["regime","baseline","feature_set","sample_id","video_id","frame_number","y_true","prob_flip","y_pred"]
    coef_fields = ["regime","feature_set","feature","coefficient_standardized","scaler_train_mean","scaler_train_scale"]
    scaler_fields = ["regime","feature_set","feature","train_mean","train_scale"]

    metrics_path = out / "metrics.csv"
    preds_path = out / "predictions.csv"
    coef_path = out / "coefficients.csv"
    scaler_path = out / "scaler_parameters.csv"
    write_dict_rows(metrics_path, metrics_rows, metric_fields)
    write_dict_rows(preds_path, prediction_rows, pred_fields)
    write_dict_rows(coef_path, coefficient_rows, coef_fields)
    write_dict_rows(scaler_path, scaler_rows, scaler_fields)

    feature_definition = {
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "resize": "aspect-ratio-preserving bilinear fit; centered zero padding",
        "scaling": "[0,1]",
        "feature_sets": FEATURE_NAMES,
        "positive_label": POSITIVE_LABEL,
        "logistic_regression": {
            "penalty": "l2", "C": 1.0, "solver": "liblinear",
            "class_weight": None, "max_iter": 2000, "random_state": 42,
            "threshold": 0.5,
        },
        "scaler": "StandardScaler fit separately on each regime's training rows only",
    }

    summary = {
        "status": "COMPLETE",
        "archive": {"bytes": archive_size, "sha256": archive_sha},
        "source_manifest_sha256": sha256_file(manifest_path),
        "split_sha256": {r: sha256_file(split_dir / f"{r}.csv") for r in ("O","S","T","ST")},
        "feature_definition": feature_definition,
        "feature_dimensions": {k: len(v) for k, v in FEATURE_NAMES.items()},
        "result_sha256": {
            "metrics.csv": sha256_file(metrics_path),
            "predictions.csv": sha256_file(preds_path),
            "coefficients.csv": sha256_file(coef_path),
            "scaler_parameters.csv": sha256_file(scaler_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_image": skimage.__version__,
        },
        "scratch_candidate_tests_examined": False,
    }
    summary_path = out / "BASELINE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Human-readable report
    metric_lookup = {(r["regime"], r["baseline"], r["feature_set"]): r for r in metrics_rows}
    lines = [
        "# MonReader V2 Baseline Results",
        "",
        "The baseline protocol was frozen before these test results were generated.",
        "",
        "| Regime | Baseline | F1 | Precision | Recall | Accuracy | Balanced acc. | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in ("O","S","T","ST"):
        for baseline, fs, label in [
            ("majority","","Majority"),
            ("logistic_regression","A","LogReg A"),
            ("logistic_regression","B","LogReg B"),
            ("logistic_regression","C","LogReg C"),
        ]:
            r = metric_lookup[(regime, baseline, fs)]
            def fmt(x):
                return "—" if x == "" else f"{float(x):.4f}"
            lines.append(
                f"| {regime} | {label} | {fmt(r['f1'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                f"{fmt(r['accuracy'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} |"
            )
    lines += [
        "",
        "All logistic-regression scaling statistics were fitted on training rows only.",
        "Feature sets A/B/C are all reported; no feature set was selected after viewing the tests.",
        "These results do not open or rank the scratch-CNN candidate bank.",
        "",
    ]
    (out / "BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print((out / "BASELINE_REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
