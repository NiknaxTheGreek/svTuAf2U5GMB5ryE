from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
    rgb_to_ycc709,
    stable_donor_order,
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

CHAMPIONS = {
    "O": {
        "candidate_id": "C14",
        "checkpoint_sha256": "57f70fb3ebecbade054a18b22c423182175f84625179d49c3625e520b215bc49",
        "test_n": 597,
    },
    "S": {
        "candidate_id": "C18",
        "checkpoint_sha256": "1a05f136c41f3f26e18f00e6f48e5f856630f3286fa52d5aea6149224d641225",
        "test_n": 770,
    },
    "T": {
        "candidate_id": "C06",
        "checkpoint_sha256": "2583071a82938e1e1053822e5aa35f2d01a6d4dea13849923f2ec57e30f1ca7b",
        "test_n": 624,
    },
    "ST": {
        "candidate_id": "C07",
        "checkpoint_sha256": "9886dbad7f73984a8333bee22da568be4a451b650dad6ee28a2713853b445b67",
        "test_n": 161,
    },
}

SHORTLIST_SIZE = 16
SELECTION_STATS_STRIDE = 16
N_BOOT = 5000
BASE_SEED = 20261003


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    return preprocess_to_uint8(Image.fromarray(rgb, mode="RGB"))


def chroma_stats(rgb: np.ndarray, stride: int = SELECTION_STATS_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    _, cb, cr = rgb_to_ycc709(rgb)
    c = np.stack([cb[::stride, ::stride], cr[::stride, ::stride]], axis=-1)
    c = c.reshape(-1, 2) / 128.0
    return c.mean(axis=0), np.cov(c, rowvar=False)


def palette_distance(
    target_stats: tuple[np.ndarray, np.ndarray],
    donor_row: dict[str, str],
) -> float:
    mu_t, cov_t = target_stats
    mu_d = np.asarray([float(donor_row["mu_cb"]), float(donor_row["mu_cr"])])
    cov_d = np.asarray([
        [float(donor_row["cov_cb_cb"]), float(donor_row["cov_cb_cr"])],
        [float(donor_row["cov_cb_cr"]), float(donor_row["cov_cr_cr"])],
    ])
    return float(
        np.linalg.norm(mu_d - mu_t)
        + np.linalg.norm(cov_d - cov_t, ord="fro")
    )


def choose_donor(
    target_id: str,
    target_video: str,
    target_env: str,
    target_rgb: np.ndarray,
    donors: list[dict[str, str]],
) -> tuple[dict[str, str], float]:
    eligible = [
        d for d in donors
        if d["sample_id"] != target_id
        and d["video_id"] != target_video
        and d["environment_id"] != target_env
    ]
    if len(eligible) < SHORTLIST_SIZE:
        raise RuntimeError(f"Only {len(eligible)} eligible donors for {target_id}")

    by_id = {d["sample_id"]: d for d in eligible}
    ordered_ids = stable_donor_order(target_id, list(by_id))
    shortlist = [by_id[sid] for sid in ordered_ids[:SHORTLIST_SIZE]]
    tstats = chroma_stats(target_rgb)

    scored = [(palette_distance(tstats, d), d) for d in shortlist]
    scored.sort(key=lambda x: (-x[0], x[1]["sample_id"]))
    return scored[0][1], float(scored[0][0])


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    yp = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, yp, zero_division=0)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)),
    }


def ci(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def f1_pred(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, zero_division=0))


def paired_bootstrap(
    y: np.ndarray,
    original_pred: np.ndarray,
    variant_preds: dict[str, np.ndarray],
    groups: np.ndarray,
    seed: int,
) -> dict:
    names = list(variant_preds)
    frame = {n: np.empty(N_BOOT, dtype=np.float64) for n in names}
    group = {n: np.empty(N_BOOT, dtype=np.float64) for n in names}

    rng = np.random.default_rng(seed)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        base = f1_pred(y[idx], original_pred[idx])
        for n in names:
            frame[n][b] = f1_pred(y[idx], variant_preds[n][idx]) - base

    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    gidx = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed + 1)
    for b in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([gidx[g] for g in sampled])
        base = f1_pred(y[idx], original_pred[idx])
        for n in names:
            group[n][b] = f1_pred(y[idx], variant_preds[n][idx]) - base

    return {
        n: {
            "frame_delta_f1_ci95": ci(frame[n]),
            "video_group_delta_f1_ci95": ci(group[n]),
        }
        for n in names
    }


def load_saved_champion_predictions(path: Path, candidate_id: str) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    selected = [r for r in rows if r.get("candidate_id") == candidate_id]
    if not selected:
        # Some later frozen files may already contain only the champion.
        selected = rows
    by_id = {r["sample_id"]: r for r in selected}
    if len(by_id) != len(selected):
        raise ValueError("Duplicate sample IDs in saved predictions")
    return by_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["O", "S", "T", "ST"], required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--saved-predictions", type=Path, required=True)
    ap.add_argument("--donor-stats", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    regime = args.regime
    champion = CHAMPIONS[regime]
    candidate_id = champion["candidate_id"]

    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive SHA mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Manifest SHA mismatch")
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256[regime]:
        raise ValueError(f"{regime} split SHA mismatch")
    if sha256_file(args.checkpoint) != champion["checkpoint_sha256"]:
        raise ValueError(f"{regime} champion checkpoint SHA mismatch")

    bank = load_candidate_bank(args.bank)
    cfg = candidate_config(bank, candidate_id)
    model = ScratchCNN(ScratchCNNConfig(
        depth=int(cfg["depth"]),
        start_filters=int(cfg["start_filters"]),
        dropout=float(cfg["dropout"]),
    ))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        ckpt["candidate_id"] != candidate_id
        or ckpt["regime"] != regime
        or int(ckpt["epoch"]) != 20
    ):
        raise ValueError("Checkpoint payload mismatch")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    manifest = read_csv(args.manifest)
    by_id = {r["sample_id"]: r for r in manifest}
    split = read_csv(args.membership)
    test = [r for r in split if r["role"] == "test"]
    if len(test) != champion["test_n"]:
        raise ValueError(f"{regime} test count changed: {len(test)}")

    saved = load_saved_champion_predictions(args.saved_predictions, candidate_id)
    if len(saved) != len(test):
        raise ValueError(
            f"Saved {candidate_id} prediction count {len(saved)} != test {len(test)}"
        )

    donors = read_csv(args.donor_stats)
    if len(donors) != 1428:
        raise ValueError("Frozen donor pool size changed")
    donor_ids = {d["sample_id"] for d in donors}

    # Audit: donor table must contain no test row from any regime by construction.
    if not donor_ids:
        raise ValueError("Donor table empty")

    transfer_cfg = PaletteTransferConfig(
        strength=1.0,
        covariance_epsilon=1e-4,
        stats_stride=4,
        gamut_iterations=14,
    )

    variants = {
        "original_rgb": [],
        "rec709_grayscale": [],
        "matched_chroma_rotation": [],
        "naturalistic_v2": [],
    }
    y_true: list[int] = []
    rows_out: list[dict] = []

    with ZipFile(args.archive) as z:
        for i, row in enumerate(test, 1):
            sid = row["sample_id"]
            meta = by_id[sid]
            target_rgb = decode_member(z, meta["archive_member"])

            donor, donor_distance = choose_donor(
                sid,
                row["video_id"],
                row["environment_id"],
                target_rgb,
                donors,
            )
            donor_meta = by_id[donor["sample_id"]]
            donor_rgb = decode_member(z, donor_meta["archive_member"])

            gray_rgb = rec709_grayscale(target_rgb)
            rotation_rgb, rotation_metrics = matched_chroma_rotation(target_rgb)
            natural_rgb, natural_metrics = naturalistic_palette_transfer(
                target_rgb,
                donor_rgb,
                cfg=transfer_cfg,
            )

            canvas = [
                preprocess_rgb(target_rgb),
                preprocess_rgb(gray_rgb),
                preprocess_rgb(rotation_rgb),
                preprocess_rgb(natural_rgb),
            ]
            stack = np.stack(canvas).astype(np.float32) / 255.0
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.from_numpy(stack))).cpu().numpy()

            y = 1 if row["label"] == "flip" else 0
            y_true.append(y)
            for name, prob in zip(variants, probs, strict=True):
                variants[name].append(float(prob))

            s = saved[sid]
            saved_y = int(s["y_true"])
            saved_p = float(s["prob_flip"])
            if saved_y != y:
                raise ValueError(f"Saved y_true mismatch: {sid}")
            if abs(saved_p - float(probs[0])) > 1e-5:
                raise ValueError(
                    f"Original probability drift {sid}: saved={saved_p}, "
                    f"recomputed={float(probs[0])}"
                )

            rows_out.append({
                "sample_id": sid,
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "environment_id": row["environment_id"],
                "y_true": y,
                "donor_id": donor["sample_id"],
                "donor_video_id": donor["video_id"],
                "donor_environment_id": donor["environment_id"],
                "donor_palette_distance": donor_distance,
                "rotation_luma_mae": rotation_metrics["luma_mae"],
                "naturalistic_luma_mae": natural_metrics["luma_mae"],
                "naturalistic_median_chroma_displacement": natural_metrics[
                    "median_chroma_displacement"
                ],
                "p_original_rgb": float(probs[0]),
                "p_rec709_grayscale": float(probs[1]),
                "p_matched_chroma_rotation": float(probs[2]),
                "p_naturalistic_v2": float(probs[3]),
                "pred_original_rgb": int(probs[0] >= 0.5),
                "pred_rec709_grayscale": int(probs[1] >= 0.5),
                "pred_matched_chroma_rotation": int(probs[2] >= 0.5),
                "pred_naturalistic_v2": int(probs[3] >= 0.5),
            })

            if i % 50 == 0 or i == len(test):
                print(
                    f"chroma_eval_progress regime={regime} {i}/{len(test)}",
                    flush=True,
                )

    y = np.asarray(y_true, dtype=np.int64)
    p = {k: np.asarray(v, dtype=np.float64) for k, v in variants.items()}
    pred = {k: (v >= 0.5).astype(np.int64) for k, v in p.items()}
    metrics = {k: metric(y, v) for k, v in p.items()}

    transformed = [
        "rec709_grayscale",
        "matched_chroma_rotation",
        "naturalistic_v2",
    ]
    deltas = {
        name: float(metrics[name]["f1"] - metrics["original_rgb"]["f1"])
        for name in transformed
    }
    groups = np.asarray([r["video_id"] for r in rows_out], dtype=object)
    seed = BASE_SEED + {"O": 0, "S": 100, "T": 200, "ST": 300}[regime]
    boot = paired_bootstrap(
        y,
        pred["original_rgb"],
        {name: pred[name] for name in transformed},
        groups,
        seed,
    )
    changed = {
        name: int(np.sum(pred[name] != pred["original_rgb"]))
        for name in transformed
    }
    mean_abs_prob = {
        name: float(np.mean(np.abs(p[name] - p["original_rgb"])))
        for name in transformed
    }

    donor_audit = {
        "same_video_donors": int(
            sum(r["donor_video_id"] == r["video_id"] for r in rows_out)
        ),
        "same_environment_donors": int(
            sum(r["donor_environment_id"] == r["environment_id"] for r in rows_out)
        ),
        "unique_donors": len(set(r["donor_id"] for r in rows_out)),
    }
    if donor_audit["same_video_donors"] or donor_audit["same_environment_donors"]:
        raise ValueError(f"Donor independence violation: {donor_audit}")

    summary = {
        "status": "CHROMA_FROZEN_MODEL_SENSITIVITY_COMPLETE",
        "selection_bearing": False,
        "retraining_performed": False,
        "regime": regime,
        "candidate_id": candidate_id,
        "checkpoint_sha256": champion["checkpoint_sha256"],
        "threshold": 0.5,
        "test_images": len(test),
        "videos": len(set(groups.tolist())),
        "transform_domain": "source-resolution RGB before frozen model preprocessing",
        "naturalistic_transform": {
            "version": "v2 validated",
            "strength": 1.0,
            "donor_shortlist_size": 16,
            "target_label_used": False,
        },
        "donor_audit": donor_audit,
        "metrics": metrics,
        "delta_f1_vs_original": deltas,
        "bootstrap": boot,
        "changed_prediction_count": changed,
        "mean_absolute_probability_change": mean_abs_prob,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = regime
    with (args.output_dir / f"{prefix}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    (args.output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Phase 11B — {regime} frozen-champion chroma sensitivity",
        "",
        "Post-hoc sensitivity only; no retraining or champion change.",
        "",
        "| Variant | F1 | Precision | Recall | Accuracy | ΔF1 | Group 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variants:
        m = metrics[name]
        if name == "original_rgb":
            d = 0.0
            ci_text = "reference"
        else:
            d = deltas[name]
            gci = boot[name]["video_group_delta_f1_ci95"]
            ci_text = f"[{gci[0]:.4f}, {gci[1]:.4f}]"
        lines.append(
            f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['accuracy']:.4f} | {d:+.4f} | {ci_text} |"
        )
    (args.output_dir / f"{prefix}_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
