from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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

from src.scratch_v2 import (
    EXPECTED_BANK_SHA256,
    EXPECTED_SPLIT_SHA256,
    MemmapDataset,
    ScratchCNN,
    ScratchCNNConfig,
    candidate_config,
    count_trainable_parameters,
    load_candidate_bank,
    sha256_file,
)


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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-data", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--cache-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-registry", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    bank = load_candidate_bank(args.bank)
    registry = json.loads(args.checkpoint_registry.read_text(encoding="utf-8"))
    if registry["status"] != "PASS_ALL_20_FROZEN_BEFORE_ST_TEST":
        raise ValueError("ST checkpoint registry did not pass the pre-test gate")
    if registry["candidate_count"] != 20 or registry["test_evaluation_opened"] is not False:
        raise ValueError("Checkpoint registry is not the closed-test 20-candidate state")

    receipt = json.loads(args.cache_receipt.read_text(encoding="utf-8"))
    if receipt["regime"] != "ST" or receipt["role"] != "test" or receipt["rows"] != 161:
        raise ValueError("ST test cache identity mismatch")
    if receipt["split_sha256"] != EXPECTED_SPLIT_SHA256["ST"]:
        raise ValueError("ST test cache split hash mismatch")
    if sha256_file(args.cache_data) != receipt["cache_data_sha256"]:
        raise ValueError("ST test cache data hash mismatch")
    if sha256_file(args.cache_index) != receipt["cache_index_sha256"]:
        raise ValueError("ST test cache index hash mismatch")

    dataset = MemmapDataset(args.cache_data, args.cache_index)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    index_rows = dataset.rows

    leaderboard = []
    predictions = []
    candidate_probabilities = {}

    # All prechecks above occur before the first model sees any ST test tensor.
    for i in range(1, 21):
        cid = f"C{i:02d}"
        config = candidate_config(bank, cid)
        record = next(row for row in registry["records"] if row["candidate_id"] == cid)
        pt = args.checkpoint_root / record["checkpoint_file"]
        if sha256_file(pt) != record["checkpoint_sha256"]:
            raise ValueError(f"{cid} checkpoint changed after freeze")

        checkpoint = torch.load(pt, map_location="cpu", weights_only=False)
        model = ScratchCNN(
            ScratchCNNConfig(
                depth=int(config["depth"]),
                start_filters=int(config["start_filters"]),
                dropout=float(config["dropout"]),
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        y_true_list = []
        probability_list = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["image"])
                probs = torch.sigmoid(logits).cpu().numpy()
                probability_list.extend(probs.tolist())
                y_true_list.extend(batch["label"].cpu().numpy().astype(int).tolist())

        y_true = np.asarray(y_true_list, dtype=np.int64)
        probability = np.asarray(probability_list, dtype=np.float64)
        if len(y_true) != 161:
            raise RuntimeError(f"{cid} did not evaluate exactly 161 ST test images")
        result = metrics(y_true, probability)
        parameters = count_trainable_parameters(model)
        leaderboard.append({
            "candidate_id": cid,
            "parameter_count": parameters,
            **result,
        })
        candidate_probabilities[cid] = probability
        pred = (probability >= 0.5).astype(np.int64)
        for row, yt, p, yp in zip(index_rows, y_true, probability, pred, strict=True):
            predictions.append({
                "candidate_id": cid,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "y_true": int(yt),
                "prob_flip": float(p),
                "y_pred": int(yp),
            })
        print(f"evaluated={cid} n=161 f1={result['f1']:.8f}", flush=True)

    leaderboard.sort(
        key=lambda row: (
            -row["f1"],
            -row["balanced_accuracy"],
            -row["pr_auc"],
            row["parameter_count"],
            row["candidate_id"],
        )
    )
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    champion = leaderboard[0]
    champion_id = champion["candidate_id"]

    # Descriptive diagnostics of the already-selected frozen Best ST model.
    y_champ = np.asarray([1 if row["label"] == "flip" else 0 for row in index_rows], dtype=np.int64)
    p_champ = candidate_probabilities[champion_id]
    best_o_candidate_on_st = next(dict(row) for row in leaderboard if row["candidate_id"] == "C14")
    best_s_candidate_on_st = next(dict(row) for row in leaderboard if row["candidate_id"] == "C18")
    best_t_candidate_on_st = next(dict(row) for row in leaderboard if row["candidate_id"] == "C06")

    per_video_rows = []
    by_video = defaultdict(list)
    for idx, row in enumerate(index_rows):
        by_video[row["video_id"]].append(idx)
    for video_id in sorted(by_video):
        inds = np.asarray(by_video[video_id], dtype=int)
        yv = y_champ[inds]
        pv = p_champ[inds]
        predv = (pv >= 0.5).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(yv, predv, labels=[0,1]).ravel()
        f1 = f1_score(yv, predv, zero_division=0)
        per_video_rows.append({
            "video_id": video_id,
            "n": len(inds),
            "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "mean_prob_flip": float(pv.mean()),
            "min_prob_flip": float(pv.min()),
            "max_prob_flip": float(pv.max()),
        })

    pred_champ = (p_champ >= 0.5).astype(np.int64)
    cases = []
    for row, yt, p, yp in zip(index_rows, y_champ, p_champ, pred_champ, strict=True):
        if yt == 0 and yp == 1:
            kind = "false_positive"
            priority = p
        elif yt == 1 and yp == 0:
            kind = "false_negative"
            priority = 1.0 - p
        else:
            kind = "low_confidence_correct"
            priority = -abs(p - 0.5)
        cases.append({
            "kind": kind,
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "frame_number": int(row["frame_number"]),
            "y_true": int(yt),
            "prob_flip": float(p),
            "y_pred": int(yp),
            "priority": float(priority),
        })
    selected_cases = []
    for kind in ("false_positive", "false_negative", "low_confidence_correct"):
        group = [row for row in cases if row["kind"] == kind]
        group.sort(key=lambda row: row["priority"], reverse=True)
        selected_cases.extend(group[:20])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "leaderboard.csv",
        leaderboard,
        ["rank","candidate_id","f1","precision","recall","accuracy","balanced_accuracy","roc_auc","pr_auc","tn","fp","fn","tp","n","parameter_count"],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        predictions,
        ["candidate_id","sample_id","video_id","frame_number","y_true","prob_flip","y_pred"],
    )
    write_csv(
        args.output_dir / "best_st_per_video.csv",
        per_video_rows,
        ["video_id","n","f1","tn","fp","fn","tp","mean_prob_flip","min_prob_flip","max_prob_flip"],
    )
    write_csv(
        args.output_dir / "best_st_error_cases.csv",
        selected_cases,
        ["kind","sample_id","video_id","frame_number","y_true","prob_flip","y_pred","priority"],
    )
    summary = {
        "status": "ST_SELECTION_BENCHMARK_COMPLETE",
        "candidate_count": 20,
        "test_rows_per_candidate": 161,
        "threshold": 0.5,
        "selection_metric": "F1",
        "tie_break_order": ["balanced_accuracy", "pr_auc", "lower_parameter_count", "candidate_id"],
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256["ST"],
        "best_st": champion,
        "best_st_candidate_id": champion_id,
        "best_o_candidate_id": "C14",
        "best_o_candidate_on_st": best_o_candidate_on_st,
        "best_s_candidate_id": "C18",
        "best_s_candidate_on_st": best_s_candidate_on_st,
        "best_t_candidate_id": "C06",
        "best_t_candidate_on_st": best_t_candidate_on_st,
        "source_temporal_safe_test_images": 161,
        "test_videos": 29,
        "train_test_video_overlap_count": 0,
        "interpretation": "Selection-benchmark estimate on the source+temporal-safe ST test; ENV-03 is absent from training and only later ENV-03 frames form the primary test; ST test selected among 20 frozen candidates.",
    }
    (args.output_dir / "ST_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "
", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
