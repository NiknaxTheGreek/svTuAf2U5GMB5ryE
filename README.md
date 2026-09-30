# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 5 — Source-Safe scratch-CNN selection benchmark: complete. O and S are frozen; T and ST remain pending.**

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
| S | C18 | **0.8196** | 0.8377 | 0.8358 | held-out ENV-03, zero train/test video overlap |

Both are **selection-benchmark estimates** because each regime test selects among 20 frozen candidates.

### Why O and S differ

The O split is highly correlated:

- every O-test image comes from a video represented in O training;
- 573/597 O-test frames have a same-video training frame only one frame number away.

S is source-disjoint:

- S training: ENV-01, ENV-02, ENV-04;
- S test: ENV-03 only;
- training videos: 88;
- test videos: 29;
- train/test video overlap: **0**.

Best S C18 achieved F1 0.8196 with 361 TN, 61 FP, 64 FN and 284 TP.

The Best-O configuration C14, retrained from scratch under the S protocol, ranked third at F1 0.6936. C18 has only 55,213 parameters versus 1,570,081 for C14.

A video-cluster bootstrap over the 29 held-out S videos gives a C18 F1 95% interval of approximately **0.674–0.912**. The paired C18 minus C14-configuration F1 difference is +0.126 with a clustered 95% interval of approximately +0.019 to +0.261. These intervals are not adjusted for selection among the 20 candidates.

See:
- `reports/data_audit/DATA_AUDIT.md`
- `reports/splits/SPLIT_AUDIT.md`
- `reports/baselines/BASELINE_REPORT.md`
- `reports/scratch/O_REPORT.md`
- `reports/scratch/S_REPORT.md`
- `manifests/scratch/O_REGISTRY.yaml`
- `manifests/scratch/S_REGISTRY.yaml`

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

Run the same frozen 20-candidate bank under **T — Temporal-Safe**. All 20 T checkpoints must be frozen and verified before the 624-image T test is opened.
