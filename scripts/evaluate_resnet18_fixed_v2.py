from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
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
from torch.utils.data import DataLoader

from src.resnet18_v2 import (
    EXPECTED_CONFIG_SHA256,
    ResNetMemmapDataset,
    build_model,
    load_fixed_config,
    verify_cache_receipt,
)
from src.scratch_v2 import EXPECTED_SPLIT_SHA256, sha256_file


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    yp = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0,1]).ravel()
    return {
        "f1": float(f1_score(y, yp, zero_division=0)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-data", required=True, type=Path)
    ap.add_argument("--cache-index", required=True, type=Path)
    ap.add_argument("--cache-receipt", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--regime", required=True, choices=("O","S","T","ST"))
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_fixed_config(args.config)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry["status"] != "PASS_ALL_4_FROZEN_BEFORE_RESNET_TESTS":
        raise ValueError("Four-checkpoint closed-test gate did not pass")
    if registry["test_evaluation_opened"] is not False:
        raise ValueError("Registry does not represent closed-test state")

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    if meta["regime"] != args.regime or meta["config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise ValueError("ResNet18 metadata identity mismatch")
    if meta["checkpoint_sha256"] != sha256_file(args.checkpoint):
        raise ValueError("ResNet18 checkpoint hash mismatch before evaluation")

    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    verify_cache_receipt(receipt, args.regime, "test", args.cache_data, args.cache_index)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["regime"] != args.regime or payload["epoch"] != 20:
        raise ValueError("ResNet18 checkpoint payload mismatch")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    dataset = ResNetMemmapDataset(args.cache_data, args.cache_index)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    yy, pp = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"]).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            pp.extend(probs.tolist())
            yy.extend(batch["label"].cpu().numpy().astype(int).tolist())

    y = np.asarray(yy, dtype=np.int64)
    p = np.asarray(pp, dtype=np.float64)
    m = metrics(y, p)
    pred = (p >= 0.5).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"RESNET18_{args.regime}_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id","video_id","frame_number","y_true","prob_flip","y_pred"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, yt, prob, yp in zip(dataset.rows, y, p, pred, strict=True):
            writer.writerow({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "y_true": int(yt),
                "prob_flip": float(prob),
                "y_pred": int(yp),
            })

    scratch_summary = json.loads(Path(f"results/scratch/{args.regime}/{args.regime}_SUMMARY.json").read_text(encoding="utf-8"))
    scratch_key = {"O":"best_o","S":"best_s","T":"best_t","ST":"best_st"}[args.regime]
    summary = {
        "status": "FIXED_RESNET18_COMPARATOR_EVALUATED",
        "regime": args.regime,
        "selection_bearing": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "threshold": 0.5,
        "metrics": m,
        "scratch_champion": scratch_summary[scratch_key],
        "f1_difference_resnet_minus_scratch_champion": float(m["f1"] - scratch_summary[scratch_key]["f1"]),
        "interpretation": "Fixed pretrained-family comparator on the identical regime test population; does not replace the frozen scratch champion.",
    }
    (args.output_dir / f"RESNET18_{args.regime}_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    (args.output_dir / f"RESNET18_{args.regime}_training_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
