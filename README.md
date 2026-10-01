# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 6 — Temporal-Safe scratch-CNN selection benchmark: complete. O, S and T are frozen; ST remains pending.**

Dataset and splits are frozen:

| Regime | Train | Primary test | Context | Excluded |
|---|---:|---:|---:|---:|
| O | 2,392 | 597 | 0 | 0 |
| S | 2,219 | 770 | 0 | 0 |
| T | 2,365 | 624 | 0 | 0 |
| ST | 1,756 | 161 | 1,072 | 0 |

The canonical C01–C20 scratch bank is frozen: 20 epochs, no validation, no early stopping, final epoch-20 checkpoint, threshold 0.5.

## Primary scratch results so far

| Regime | Champion | F1 | Accuracy | Balanced acc. | Interpretation |
|---|---|---:|---:|---:|---|
| O | C14 | **0.9983** | 0.9983 | 0.9983 | supplied source-joint benchmark |
| S | C18 | **0.8196** | 0.8377 | 0.8358 | held-out ENV-03 |
| T | C06 | **0.8800** | 0.8894 | 0.8900 | later frames of the same videos |
| ST | pending | — | — | — | held-out source + temporal shift |

All completed scratch results are **selection-benchmark estimates** because each regime test selects among 20 candidates frozen before that test opens.

## Key interpretation

O is extremely easy relative to the stricter regimes: every O-test image comes from a training-seen video and 573/597 O-test frames have a same-video training frame only one frame number away.

S removes source overlap and falls to F1 0.8196.

T preserves source identity but forces chronological separation and reaches F1 0.8800 with C06. A video-cluster bootstrap gives C06 F1 approximately **0.837–0.918**.

The O-winning C14 configuration behaves very differently under T. Retrained on T training data, it has F1 0.2981 at threshold 0.5 but ROC-AUC 0.9414 and PR-AUC 0.9567. That pattern indicates substantial operating-threshold/calibration shift rather than disappearance of discriminative ranking signal. The frozen primary protocol does not permit post-test threshold tuning.

See:
- `reports/data_audit/DATA_AUDIT.md`
- `reports/splits/SPLIT_AUDIT.md`
- `reports/baselines/BASELINE_REPORT.md`
- `reports/scratch/O_REPORT.md`
- `reports/scratch/S_REPORT.md`
- `reports/scratch/T_REPORT.md`
- `manifests/scratch/O_REGISTRY.yaml`
- `manifests/scratch/S_REGISTRY.yaml`
- `manifests/scratch/T_REGISTRY.yaml`

## Frozen scratch-CNN rules

- same C01–C20 bank in O/S/T/ST;
- exactly 20 epochs;
- BCEWithLogitsLoss;
- natural class distribution;
- no primary augmentation;
- no validation / early stopping;
- final epoch-20 checkpoint;
- threshold 0.5;
- all candidates in a regime are frozen before that regime test opens;
- champion order: F1 → balanced accuracy → PR-AUC → lower parameter count → candidate ID.

## Next gate

Run **ST — Source + Temporal Safe** with the same frozen bank. All 20 ST checkpoints must be frozen and verified before the 161-image ST primary test is opened. The 1,072 ST context rows are not part of the primary test score.
