# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 4 — Original scratch-CNN selection benchmark: complete. Source-safe/temporal scratch regimes remain pending.**

Dataset and splits are frozen:

| Regime | Train | Primary test | Context | Excluded |
|---|---:|---:|---:|---:|
| O | 2,392 | 597 | 0 | 0 |
| S | 2,219 | 770 | 0 | 0 |
| T | 2,365 | 624 | 0 | 0 |
| ST | 1,756 | 161 | 1,072 | 0 |

The canonical C01–C20 scratch bank is frozen: 20 epochs, no validation, no early stopping, final epoch-20 checkpoint, threshold 0.5.

## Original benchmark — Best O

All 20 O candidates were trained and frozen before the O test was opened. The pre-test checkpoint gate passed for every candidate.

**Best O = C14**

| Metric | C14 |
|---|---:|
| F1 | **0.9983** |
| Precision | 1.0000 |
| Recall | 0.9966 |
| Accuracy | 0.9983 |
| Balanced accuracy | 0.9983 |
| ROC-AUC | 0.9998 |
| PR-AUC | 0.9998 |
| Errors | 1 / 597 |

This is a **selection-benchmark estimate** because O test F1 selected among 20 frozen candidates.

The high score requires careful interpretation:

- exact decoded-pixel duplicate groups: 0;
- every O-test image comes from a video represented in O training;
- 573/597 O-test frames have a same-video training frame only one frame number away;
- no source-disjoint Original diagnostic exists.

The sole C14 error was a false negative. Visual review showed it was surrounded by extremely similar same-video training frames, directly illustrating the strong temporal correlation in the supplied split.

The 20-image O temporal-future subset is perfect for C14, but it remains source-joint.

See:
- `reports/data_audit/DATA_AUDIT.md`
- `reports/splits/SPLIT_AUDIT.md`
- `reports/baselines/BASELINE_REPORT.md`
- `reports/scratch/O_REPORT.md`
- `manifests/scratch/O_REGISTRY.yaml`

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

Run the same frozen 20 candidates under **S — Source-Safe**, where ENV-03 is absent from training. S is the first primary test of genuinely unseen-source/environment behavior.
