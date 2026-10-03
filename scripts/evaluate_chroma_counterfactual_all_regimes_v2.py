from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
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

from src.chroma_counterfactual_v2 import (
    PaletteTransferConfig,
    matched_chroma_rotation,
    naturalistic_palette_transfer,
    rec709_grayscale,
)
from src.scratch_v2 import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    ScratchCNN,
    ScratchCNNConfig,
    candidate_config,
    load_candidate_bank,
    preprocess_to_uint8,
    read_csv,
    sha256_file,
)
from scripts.validate_naturalistic_chroma_v2 import (
    choose_donor_v2,
    decode_member,
)

CHAMPIONS = {
    "O": {
        "candidate_id": "C14",
        "checkpoint_sha256": "57f70fb3ebecbade054a18b22c423182175f84625179d49c3625e520b215bc49",
        "checkpoint_file": "BestO.pt",
        "test_count": 597,
    },
    "S": {
        "candidate_id": "C18",
        "checkpoint_sha256": "1a05f136c41f3f26e18f00e6f48e5f856630f3286fa52d5aea6149224d641225",
        "checkpoint_file": "BestS.pt",
        "test_count": 770,
    },
    "T": {
        "candidate_id": "C06",
        "checkpoint_sha256": "2583071a82938e1e1053822e5aa35f2d01a6d4dea13849923f2ec57e30f1ca7b",
        "checkpoint_file": "BestT.pt",
        "test_count": 624,
    },
    "ST": {
        "candidate_id": "C07",
        "checkpoint_sha256": "9886dbad7f73984a8333bee22da568be4a451b650dad6ee28a2713853b445b67",
        "checkpoint_file": "BestST.pt",
        "test_count": 161,
    },
}

VARIANTS = ("original", "grayscale", "matched_rotation", "naturalistic_v2")
N_BOOT = 5000
SEED = 20261003


def metric(y: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)),
    }


def ci(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def f1_binary(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, zero_division=0))


def paired_bootstrap(
    y: np.ndarray,
    pred_original: np.ndarray,
    transformed: dict[str, np.ndarray],
    groups: np.ndarray,
) -> dict:
    names = list(transformed)
    frame = {n: np.empty(N_BOOT, dtype=np.float64) for n in names}
    group = {n: np.empty(N_BOOT, dtype=np.float64) for n in names}
    contrast_frame = np.empty(N_BOOT, dtype=np.float64)
    contrast_group = np.empty(N_BOOT, dtype=np.float64)

    rng = np.random.default_rng(SEED)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        base = f1_binary(y[idx], pred_original[idx])
        vals = {}
        for n in names:
            vals[n] = f1_binary(y[idx], transformed[n][idx]) - base
            frame[n][b] = vals[n]
        contrast_frame[b] = vals["naturalistic_v2"] - vals["matched_rotation"]

    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    group_indices = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(SEED + 1)
    for b in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        base = f1_binary(y[idx], pred_original[idx])
        vals = {}
        for n in names:
            vals[n] = f1_binary(y[idx], transformed[n][idx]) - base
            group[n][b] = vals[n]
        contrast_group[b] = vals["naturalistic_v2"] - vals["matched_rotation"]

    out = {
        n: {
            "frame_delta_f1_ci95": ci(frame[n]),
            "video_group_delta_f1_ci95": ci(group[n]),
        }
        for n in names
    }
    out["naturalistic_minus_rotation_delta_f1_contrast"] = {
        "frame_ci95": ci(contrast_frame),
        "video_group_ci95": ci(contrast_group),
    }
    return out


def load_saved_champion_predictions(
    path: Path,
    candidate_id: str,
) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["candidate_id"] == candidate_id:
                rows[row["sample_id"]] = row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=["O", "S", "T", "ST"])
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--membership", required=True, type=Path)
    ap.add_argument("--split-o", required=True, type=Path)
    ap.add_argument("--split-s", required=True, type=Path)
    ap.add_argument("--split-t", required=True, type=Path)
    ap.add_argument("--split-st", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--evidence-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    spec = CHAMPIONS[args.regime]
    cid = spec["candidate_id"]
    checkpoint = args.evidence_root / spec["checkpoint_file"]
    saved_predictions_path = args.evidence_root / "predictions.csv"

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive SHA mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("manifest SHA mismatch")
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256[args.regime]:
        raise ValueError(f"{args.regime} split SHA mismatch")
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError(f"{args.regime} champion checkpoint SHA mismatch")
    if not saved_predictions_path.exists():
        raise FileNotFoundError(saved_predictions_path)

    bank = load_candidate_bank(args.bank)
    cfg_model = candidate_config(bank, cid)
    model = ScratchCNN(
        ScratchCNNConfig(
            depth=int(cfg_model["depth"]),
            start_filters=int(cfg_model["start_filters"]),
            dropout=float(cfg_model["dropout"]),
        )
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["candidate_id"] != cid or ckpt["regime"] != args.regime or ckpt["epoch"] != 20:
        raise ValueError("checkpoint payload identity mismatch")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    membership = read_csv(args.membership)
    test_rows = [r for r in membership if r["role"] == "test"]
    if len(test_rows) != spec["test_count"]:
        raise ValueError(
            f"{args.regime} expected {spec['test_count']} test rows, got {len(test_rows)}"
        )

    all_splits = {
        "O": read_csv(args.split_o),
        "S": read_csv(args.split_s),
        "T": read_csv(args.split_t),
        "ST": read_csv(args.split_st),
    }
    roles = {
        regime: {r["sample_id"]: r["role"] for r in rows}
        for regime, rows in all_splits.items()
    }
    env_by_id: dict[str, str] = {}
    for rows in all_splits.values():
        for r in rows:
            env_by_id.setdefault(r["sample_id"], r["environment_id"])

    donor_ids = sorted(
        sid for sid in by_id
        if all(roles[r].get(sid) != "test" for r in ("O", "S", "T", "ST"))
    )
    if len(donor_ids) != 1428:
        raise ValueError(f"universal donor pool changed: {len(donor_ids)}")

    saved = load_saved_champion_predictions(saved_predictions_path, cid)
    if len(saved) != spec["test_count"]:
        raise ValueError(
            f"saved {cid} predictions count mismatch: {len(saved)}"
        )

    transfer_cfg = PaletteTransferConfig(
        strength=1.0,
        covariance_epsilon=1e-4,
        stats_stride=4,
        gamut_iterations=14,
    )

    probabilities = {name: [] for name in VARIANTS}
    rows_out: list[dict] = []
    y_true_list: list[int] = []
    donor_stats_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with ZipFile(args.archive) as z:
        for k, row in enumerate(test_rows, 1):
            sid = row["sample_id"]
            src = by_id[sid]
            source_rgb = decode_member(z, src["archive_member"])

            donor_id, donor_distance, shortlist = choose_donor_v2(
                sid,
                src["video_id"],
                row["environment_id"],
                donor_ids,
                by_id,
                env_by_id,
                source_rgb,
                z,
                donor_stats_cache,
            )
            donor_rgb = decode_member(z, by_id[donor_id]["archive_member"])

            gray_rgb = rec709_grayscale(source_rgb)
            rotation_rgb, rotation_diag = matched_chroma_rotation(
                source_rgb, iterations=transfer_cfg.gamut_iterations
            )
            natural_rgb, natural_diag = naturalistic_palette_transfer(
                source_rgb, donor_rgb, cfg=transfer_cfg
            )

            with Image.fromarray(source_rgb, mode="RGB") as im:
                original = preprocess_to_uint8(im)
            with Image.fromarray(gray_rgb, mode="RGB") as im:
                grayscale = preprocess_to_uint8(im)
            with Image.fromarray(rotation_rgb, mode="RGB") as im:
                rotation = preprocess_to_uint8(im)
            with Image.fromarray(natural_rgb, mode="RGB") as im:
                natural = preprocess_to_uint8(im)

            stack = np.stack(
                [original, grayscale, rotation, natural],
                axis=0,
            ).astype(np.float32) / 255.0

            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.from_numpy(stack))
                ).cpu().numpy()

            y = 1 if row["label"] == "flip" else 0
            y_true_list.append(y)
            for name, prob in zip(VARIANTS, probs, strict=True):
                probabilities[name].append(float(prob))

            saved_row = saved[sid]
            if int(saved_row["y_true"]) != y:
                raise ValueError(f"saved y_true drift: {sid}")
            if abs(float(saved_row["prob_flip"]) - float(probs[0])) > 1e-5:
                raise ValueError(
                    f"original probability drift {sid}: "
                    f"{saved_row['prob_flip']} vs {float(probs[0])}"
                )

            rows_out.append({
                "sample_id": sid,
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "environment_id": row["environment_id"],
                "y_true": y,
                "donor_id": donor_id,
                "donor_video_id": by_id[donor_id]["video_id"],
                "donor_environment_id": env_by_id.get(donor_id, ""),
                "donor_palette_distance": float(donor_distance),
                "donor_shortlist_size": len(shortlist),
                "naturalistic_luma_mae": float(natural_diag["luma_mae"]),
                "naturalistic_median_chroma_displacement": float(
                    natural_diag["median_chroma_displacement"]
                ),
                "naturalistic_gamut_compressed_fraction": float(
                    natural_diag["gamut_compressed_fraction"]
                ),
                "rotation_luma_mae": float(rotation_diag["luma_mae"]),
                "rotation_gamut_compressed_fraction": float(
                    rotation_diag["gamut_compressed_fraction"]
                ),
                "p_original": float(probs[0]),
                "p_grayscale": float(probs[1]),
                "p_matched_rotation": float(probs[2]),
                "p_naturalistic_v2": float(probs[3]),
                "pred_original": int(probs[0] >= 0.5),
                "pred_grayscale": int(probs[1] >= 0.5),
                "pred_matched_rotation": int(probs[2] >= 0.5),
                "pred_naturalistic_v2": int(probs[3] >= 0.5),
            })

            if k % 50 == 0 or k == len(test_rows):
                print(
                    f"regime={args.regime} chroma_progress={k}/{len(test_rows)}",
                    flush=True,
                )

    y = np.asarray(y_true_list, dtype=np.int64)
    p = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in probabilities.items()
    }
    pred = {name: (values >= 0.5).astype(np.int64) for name, values in p.items()}
    metrics = {name: metric(y, values) for name, values in p.items()}

    transformed_names = ("grayscale", "matched_rotation", "naturalistic_v2")
    delta_f1 = {
        name: float(metrics[name]["f1"] - metrics["original"]["f1"])
        for name in transformed_names
    }
    groups = np.asarray([r["video_id"] for r in rows_out], dtype=object)
    bootstrap = paired_bootstrap(
        y,
        pred["original"],
        {name: pred[name] for name in transformed_names},
        groups,
    )
    changed_prediction_count = {
        name: int(np.sum(pred[name] != pred["original"]))
        for name in transformed_names
    }
    mean_abs_probability_change = {
        name: float(np.mean(np.abs(p[name] - p["original"])))
        for name in transformed_names
    }

    natural_luma = np.asarray(
        [r["naturalistic_luma_mae"] for r in rows_out], dtype=np.float64
    )
    natural_dc = np.asarray(
        [r["naturalistic_median_chroma_displacement"] for r in rows_out],
        dtype=np.float64,
    )
    rotation_luma = np.asarray(
        [r["rotation_luma_mae"] for r in rows_out], dtype=np.float64
    )

    summary = {
        "status": "CHROMA_FROZEN_CHAMPION_EVALUATION_COMPLETE",
        "selection_bearing": False,
        "retraining_performed": False,
        "regime": args.regime,
        "candidate_id": cid,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "split_sha256": EXPECTED_SPLIT_SHA256[args.regime],
        "test_images": len(test_rows),
        "videos": len(set(groups.tolist())),
        "threshold": 0.5,
        "variants": list(VARIANTS),
        "metrics": metrics,
        "delta_f1_vs_original": delta_f1,
        "bootstrap": bootstrap,
        "changed_prediction_count": changed_prediction_count,
        "mean_absolute_probability_change": mean_abs_probability_change,
        "transform_diagnostics": {
            "universal_donor_pool_size": len(donor_ids),
            "same_video_donors": int(
                sum(r["donor_video_id"] == r["video_id"] for r in rows_out)
            ),
            "same_environment_donors": int(
                sum(r["donor_environment_id"] == r["environment_id"] for r in rows_out)
            ),
            "naturalistic_mean_luma_mae": float(natural_luma.mean()),
            "naturalistic_p95_luma_mae": float(
                np.quantile(natural_luma, 0.95)
            ),
            "naturalistic_median_chroma_displacement": float(
                np.median(natural_dc)
            ),
            "rotation_mean_luma_mae": float(rotation_luma.mean()),
        },
        "interpretation_limit": (
            "Post-hoc frozen-model sensitivity only. "
            "No result may replace the frozen O/S/T/ST champions."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.regime}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    changed_rows = [
        r for r in rows_out
        if (
            r["pred_grayscale"] != r["pred_original"]
            or r["pred_matched_rotation"] != r["pred_original"]
            or r["pred_naturalistic_v2"] != r["pred_original"]
        )
    ]
    with (args.output_dir / f"{args.regime}_changed_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(changed_rows)

    (args.output_dir / f"{args.regime}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
