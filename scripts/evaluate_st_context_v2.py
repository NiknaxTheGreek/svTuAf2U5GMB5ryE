from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from src.scratch_v2 import (
    EXPECTED_BANK_SHA256, EXPECTED_SPLIT_SHA256, MemmapDataset,
    ScratchCNN, ScratchCNNConfig, candidate_config, load_candidate_bank,
    sha256_file,
)

BEST_ST_ID = "C07"
BEST_ST_SHA256 = "9886dbad7f73984a8333bee22da568be4a451b650dad6ee28a2713853b445b67"


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def load_population(data_path: Path, index_path: Path, receipt_path: Path, expected_detail: str, expected_rows: int):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["regime"] != "ST" or receipt["role"] != "context":
        raise ValueError("Context receipt regime/role mismatch")
    if receipt["role_detail"] != expected_detail or receipt["rows"] != expected_rows:
        raise ValueError("Context receipt population mismatch")
    if receipt["split_sha256"] != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("Context split SHA mismatch")
    if sha256_file(data_path) != receipt["cache_data_sha256"]:
        raise ValueError("Context cache data hash mismatch")
    if sha256_file(index_path) != receipt["cache_index_sha256"]:
        raise ValueError("Context cache index hash mismatch")
    return MemmapDataset(data_path, index_path)


def evaluate_dataset(model: ScratchCNN, dataset: MemmapDataset, population: str):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    y, p = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"])).cpu().numpy()
            p.extend(probs.tolist())
            y.extend(batch["label"].cpu().numpy().astype(int).tolist())
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    rows = []
    for src, yt, prob, yp in zip(dataset.rows, y, p, pred, strict=True):
        rows.append({
            "population": population,
            "sample_id": src["sample_id"],
            "video_id": src["video_id"],
            "frame_number": int(src["frame_number"]),
            "environment_id": src["environment_id"],
            "y_true": int(yt),
            "prob_flip": float(prob),
            "y_pred": int(yp),
        })
    return metrics(y, p), rows


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--unseen-data", required=True, type=Path)
    ap.add_argument("--unseen-index", required=True, type=Path)
    ap.add_argument("--unseen-receipt", required=True, type=Path)
    ap.add_argument("--known-data", required=True, type=Path)
    ap.add_argument("--known-index", required=True, type=Path)
    ap.add_argument("--known-receipt", required=True, type=Path)
    ap.add_argument("--primary-predictions", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    bank = load_candidate_bank(args.bank)
    cfg = candidate_config(bank, BEST_ST_ID)
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    if meta["candidate_id"] != BEST_ST_ID or meta["regime"] != "ST":
        raise ValueError("Best ST metadata identity mismatch")
    if meta["checkpoint_sha256"] != BEST_ST_SHA256 or sha256_file(args.checkpoint) != BEST_ST_SHA256:
        raise ValueError("Frozen Best ST checkpoint SHA mismatch")
    if meta["split_sha256"] != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("Best ST metadata split mismatch")
    if meta["bank_sha256"] != EXPECTED_BANK_SHA256:
        raise ValueError("Best ST candidate-bank mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["candidate_id"] != BEST_ST_ID or checkpoint["regime"] != "ST" or checkpoint["epoch"] != 20:
        raise ValueError("Best ST checkpoint payload identity mismatch")
    model = ScratchCNN(ScratchCNNConfig(
        depth=int(cfg["depth"]), start_filters=int(cfg["start_filters"]), dropout=float(cfg["dropout"])
    ))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    unseen = load_population(
        args.unseen_data, args.unseen_index, args.unseen_receipt,
        "heldout_environment_earlier_context", 609,
    )
    known = load_population(
        args.known_data, args.known_index, args.known_receipt,
        "known_environment_later_context", 463,
    )

    unseen_metrics, unseen_rows = evaluate_dataset(model, unseen, "context_unseen_early")
    known_metrics, known_rows = evaluate_dataset(model, known, "context_known_future")

    with args.primary_predictions.open(newline="", encoding="utf-8") as handle:
        all_primary = list(csv.DictReader(handle))
    primary = [r for r in all_primary if r["candidate_id"] == BEST_ST_ID]
    if len(primary) != 161:
        raise ValueError("Saved C07 ST primary predictions must contain 161 rows")
    late_y = np.asarray([int(r["y_true"]) for r in primary], dtype=np.int64)
    late_p = np.asarray([float(r["prob_flip"]) for r in primary], dtype=np.float64)

    early_y = np.asarray([r["y_true"] for r in unseen_rows], dtype=np.int64)
    early_p = np.asarray([r["prob_flip"] for r in unseen_rows], dtype=np.float64)
    combined_metrics = metrics(np.concatenate([early_y, late_y]), np.concatenate([early_p, late_p]))

    per_video = []
    for population, rows in (("context_unseen_early", unseen_rows), ("context_known_future", known_rows)):
        groups = defaultdict(list)
        for row in rows:
            groups[row["video_id"]].append(row)
        for video_id in sorted(groups):
            group = groups[video_id]
            y = np.asarray([r["y_true"] for r in group], dtype=np.int64)
            p = np.asarray([r["prob_flip"] for r in group], dtype=np.float64)
            m = metrics(y, p) if len(set(y.tolist())) > 1 else None
            pred = (p >= 0.5).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            per_video.append({
                "population": population, "video_id": video_id, "n": len(group),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "mean_prob_flip": float(p.mean()),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "context_predictions.csv",
        unseen_rows + known_rows,
        ["population","sample_id","video_id","frame_number","environment_id","y_true","prob_flip","y_pred"],
    )
    write_csv(
        args.output_dir / "context_per_video.csv",
        per_video,
        ["population","video_id","n","f1","tn","fp","fn","tp","mean_prob_flip"],
    )
    summary = {
        "status": "ST_CONTEXT_DIAGNOSTICS_COMPLETE",
        "candidate_id": BEST_ST_ID,
        "checkpoint_sha256": BEST_ST_SHA256,
        "selection_bearing": False,
        "retraining_performed": False,
        "primary_champion_changed": False,
        "context_unseen_early": unseen_metrics,
        "context_known_future": known_metrics,
        "post_hoc_combined_ENV03_source_safe_view": {
            **combined_metrics,
            "rows": 770,
            "composition": "609 earlier ENV-03 context rows + 161 saved ST primary later ENV-03 predictions",
            "note": "Post-hoc descriptive view only; it does not redefine S or ST."
        }
    }
    (args.output_dir / "ST_CONTEXT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
