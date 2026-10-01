from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

SEED = 20261001
N_BOOT = 5000
FIXED_O_CANDIDATE = "C14"

REGIMES = {
    "O": {
        "champion": "C14",
        "n": 597,
        "summary_key": "best_o",
        "question": "supplied source-joint benchmark",
    },
    "S": {
        "champion": "C18",
        "n": 770,
        "summary_key": "best_s",
        "question": "unseen ENV-03 source/environment",
    },
    "T": {
        "champion": "C06",
        "n": 624,
        "summary_key": "best_t",
        "question": "later frames within known videos/sources",
    },
    "ST": {
        "champion": "C07",
        "n": 161,
        "summary_key": "best_st",
        "question": "unseen ENV-03 plus later-frame stress test",
    },
}


def load_summary(regime: str) -> dict:
    path = Path(f"results/scratch/{regime}/{regime}_SUMMARY.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = REGIMES[regime]
    if payload[spec["summary_key"]]["candidate_id"] != spec["champion"]:
        raise ValueError(f"{regime} champion identity changed")
    if payload[spec["summary_key"]]["n"] != spec["n"]:
        raise ValueError(f"{regime} primary test size changed")
    return payload


def load_predictions(regime: str, candidate_id: str) -> dict[str, np.ndarray]:
    path = Path(f"results/scratch/{regime}/predictions.csv")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["candidate_id"] == candidate_id:
                rows.append(row)
    expected = REGIMES[regime]["n"]
    if len(rows) != expected:
        raise ValueError(f"{regime}/{candidate_id} has {len(rows)} rows, expected {expected}")
    y = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int64)
    p = np.asarray([float(r["prob_flip"]) for r in rows], dtype=np.float64)
    yp_saved = np.asarray([int(r["y_pred"]) for r in rows], dtype=np.int64)
    yp = (p >= 0.5).astype(np.int64)
    if not np.array_equal(yp, yp_saved):
        raise ValueError(f"{regime}/{candidate_id} saved y_pred differs from threshold 0.5")
    return {
        "y": y,
        "p": p,
        "yp": yp,
        "video": np.asarray([r["video_id"] for r in rows], dtype=object),
    }


def full_metrics(y: np.ndarray, p: np.ndarray) -> dict:
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y)),
    }


def bootstrap_metrics(y: np.ndarray, yp: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, float, float]:
    yt = y[indices]
    pt = yp[indices]
    tp = int(np.sum((yt == 1) & (pt == 1)))
    fp = int(np.sum((yt == 0) & (pt == 1)))
    fn = int(np.sum((yt == 1) & (pt == 0)))
    tn = int(np.sum((yt == 0) & (pt == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / len(indices)
    tpr = recall if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    bacc = (tpr + tnr) / 2 if np.isfinite(tpr) and np.isfinite(tnr) else np.nan
    return f1, precision, recall, accuracy, bacc


def percentile(values: np.ndarray) -> list[float]:
    clean = values[np.isfinite(values)]
    return [float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))]


def bootstrap_frame(y: np.ndarray, yp: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(y)
    vals = np.empty((N_BOOT, 5), dtype=np.float64)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals[i] = bootstrap_metrics(y, yp, idx)
    return {
        "f1": percentile(vals[:, 0]),
        "precision": percentile(vals[:, 1]),
        "recall": percentile(vals[:, 2]),
        "accuracy": percentile(vals[:, 3]),
        "balanced_accuracy": percentile(vals[:, 4]),
    }


def bootstrap_group(y: np.ndarray, yp: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict:
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    group_indices = {g: np.where(groups == g)[0] for g in unique}
    vals = np.empty((N_BOOT, 5), dtype=np.float64)
    for i in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        vals[i] = bootstrap_metrics(y, yp, idx)
    return {
        "f1": percentile(vals[:, 0]),
        "precision": percentile(vals[:, 1]),
        "recall": percentile(vals[:, 2]),
        "accuracy": percentile(vals[:, 3]),
        "balanced_accuracy": percentile(vals[:, 4]),
        "groups": int(len(unique)),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    out = Path("results/synthesis")
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/synthesis")
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = {r: load_summary(r) for r in REGIMES}
    champion_rows = []
    fixed_rows = []
    uncertainty = {
        "method": "percentile_bootstrap",
        "frame_resamples": N_BOOT,
        "video_group_resamples": N_BOOT,
        "seed": SEED,
        "selection_conditioned": True,
        "note": "Intervals condition on the champion selected from the frozen 20-candidate bank on the same regime test; they are not selection-adjusted.",
        "regimes": {},
    }

    for offset, (regime, spec) in enumerate(REGIMES.items()):
        champ = load_predictions(regime, spec["champion"])
        observed = full_metrics(champ["y"], champ["p"])
        saved = summaries[regime][spec["summary_key"]]
        for key in ("f1", "precision", "recall", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc"):
            if not np.isclose(observed[key], float(saved[key]), rtol=0, atol=1e-12):
                raise ValueError(f"{regime} recomputed {key} differs from saved summary")

        frame_rng = np.random.default_rng(SEED + offset * 100 + 1)
        group_rng = np.random.default_rng(SEED + offset * 100 + 2)
        frame_ci = bootstrap_frame(champ["y"], champ["yp"], frame_rng)
        group_ci = bootstrap_group(champ["y"], champ["yp"], champ["video"], group_rng)
        uncertainty["regimes"][regime] = {
            "champion": spec["champion"],
            "observed": observed,
            "frame_bootstrap_ci95": frame_ci,
            "video_group_bootstrap_ci95": group_ci,
        }
        champion_rows.append({
            "regime": regime,
            "champion": spec["champion"],
            "test_n": spec["n"],
            "video_groups": group_ci["groups"],
            "question": spec["question"],
            **observed,
            "frame_f1_ci_low": frame_ci["f1"][0],
            "frame_f1_ci_high": frame_ci["f1"][1],
            "group_f1_ci_low": group_ci["f1"][0],
            "group_f1_ci_high": group_ci["f1"][1],
        })

        fixed = load_predictions(regime, FIXED_O_CANDIDATE)
        fixed_m = full_metrics(fixed["y"], fixed["p"])
        fixed_rows.append({
            "regime": regime,
            "candidate_id": FIXED_O_CANDIDATE,
            "test_n": spec["n"],
            "question": spec["question"],
            **fixed_m,
        })

    base_f1 = fixed_rows[0]["f1"]
    for row in fixed_rows:
        row["f1_retention_vs_O"] = float(row["f1"] / base_f1) if base_f1 else np.nan
        row["f1_change_vs_O"] = float(row["f1"] - base_f1)

    champion_fields = [
        "regime","champion","test_n","video_groups","question",
        "f1","precision","recall","accuracy","balanced_accuracy","roc_auc","pr_auc",
        "tn","fp","fn","tp","n",
        "frame_f1_ci_low","frame_f1_ci_high","group_f1_ci_low","group_f1_ci_high",
    ]
    fixed_fields = [
        "regime","candidate_id","test_n","question",
        "f1","precision","recall","accuracy","balanced_accuracy","roc_auc","pr_auc",
        "tn","fp","fn","tp","n","f1_retention_vs_O","f1_change_vs_O",
    ]
    write_csv(out / "champion_envelope.csv", champion_rows, champion_fields)
    write_csv(out / "fixed_C14_robustness.csv", fixed_rows, fixed_fields)
    (out / "bootstrap_uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Figure 1: champion F1 with frame and group CIs.
    labels = [r["regime"] for r in champion_rows]
    x = np.arange(len(labels), dtype=float)
    y = np.asarray([r["f1"] for r in champion_rows])
    flo = np.asarray([r["frame_f1_ci_low"] for r in champion_rows])
    fhi = np.asarray([r["frame_f1_ci_high"] for r in champion_rows])
    glo = np.asarray([r["group_f1_ci_low"] for r in champion_rows])
    ghi = np.asarray([r["group_f1_ci_high"] for r in champion_rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x - 0.05, y, yerr=np.vstack([y-flo, fhi-y]), fmt="o", capsize=4, label="Frame bootstrap 95% CI")
    ax.errorbar(x + 0.05, y, yerr=np.vstack([y-glo, ghi-y]), fmt="s", capsize=4, label="Video/group bootstrap 95% CI")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("MonReader V2 scratch champion envelope")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "champion_f1_ci.svg")
    plt.close(fig)

    # Figure 2: fixed O-winning configuration C14 across regimes.
    fy = np.asarray([r["f1"] for r in fixed_rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, fy, marker="o")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 at threshold 0.5")
    ax.set_title("Fixed C14 configuration retrained within each regime")
    for xx, yy in zip(x, fy, strict=True):
        ax.text(xx, min(1.03, yy + 0.035), f"{yy:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "fixed_C14_f1.svg")
    plt.close(fig)

    md = [
        "# MonReader V2 — Cross-Regime Synthesis",
        "",
        "## Scope",
        "",
        "This phase synthesizes already-frozen scratch-CNN results. It performs no training, no threshold tuning, and no new model selection. All uncertainty is explicitly selection-conditioned.",
        "",
        "## Champion envelope",
        "",
        "| Regime | Champion | n | F1 | Frame 95% CI | Video/group 95% CI | Accuracy | ROC-AUC | PR-AUC | Question |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in champion_rows:
        md.append(
            f"| {r['regime']} | {r['champion']} | {r['test_n']} | **{r['f1']:.4f}** | "
            f"{r['frame_f1_ci_low']:.3f}–{r['frame_f1_ci_high']:.3f} | "
            f"{r['group_f1_ci_low']:.3f}–{r['group_f1_ci_high']:.3f} | "
            f"{r['accuracy']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['question']} |"
        )
    md += [
        "",
        "The regimes are not interchangeable evaluation populations, so the numerically highest F1 must not be called the globally best model. O is the conventional supplied benchmark; S isolates unseen-source difficulty; T isolates later-frame difficulty within known sources; ST combines unseen source and later-frame stress.",
        "",
        "## Fixed O-winning configuration: C14",
        "",
        "C14 is retrained from scratch inside each regime using the same frozen architecture/hyperparameters; these are not the same learned weights.",
        "",
        "| Regime | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC | F1 retention vs O |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fixed_rows:
        md.append(
            f"| {r['regime']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['accuracy']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1_retention_vs_O']:.3f} |"
        )
    md += [
        "",
        "C14's fixed-threshold F1 degrades sharply outside O, especially in T and ST. In T, high ROC-AUC/PR-AUC coexist with low F1, consistent with a large operating-threshold/calibration shift rather than total loss of ranking signal.",
        "",
        "## Uncertainty",
        "",
        f"Each champion uses {N_BOOT:,} frame bootstrap resamples and {N_BOOT:,} canonical-video/group bootstrap resamples with fixed seed {SEED}. The intervals are not adjusted for selecting the champion from 20 candidates on the same test.",
        "",
        "## Next gate",
        "",
        "Proceed to the fixed ImageNet ResNet18 comparator. It must remain separate from scratch champion selection: no sweep, no validation, 5 fixed head-only epochs + 15 fixed full-network epochs, final epoch-20 checkpoint, and one test evaluation per selected regime.",
        "",
    ]
    (report_dir / "CROSS_REGIME_SYNTHESIS.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "status": "PHASE_9_SYNTHESIS_COMPLETE",
        "frame_bootstrap_resamples": N_BOOT,
        "video_group_bootstrap_resamples": N_BOOT,
        "seed": SEED,
        "selection_conditioned": True,
        "champions": {r["regime"]: r["champion"] for r in champion_rows},
        "champion_f1": {r["regime"]: r["f1"] for r in champion_rows},
        "fixed_o_candidate": FIXED_O_CANDIDATE,
        "next_gate": "fixed_resnet18_comparator",
    }
    (out / "SYNTHESIS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
