# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Primary scratch-CNN execution is complete for O, S, T and ST. Phase 9 cross-regime synthesis is next.**

| Regime | Champion | F1 | Accuracy | Balanced acc. | Interpretation |
|---|---:|---:|---:|---:|---|
| O | C14 | **0.9983** | 0.9983 | 0.9983 | supplied source-joint benchmark |
| S | C18 | **0.8196** | 0.8377 | 0.8358 | held-out ENV-03 |
| T | C06 | **0.8800** | 0.8894 | 0.8900 | later frames of the same videos |
| ST | C07 | **0.7040** | 0.7702 | 0.7601 | held-out ENV-03 + later frames |

These are selection-benchmark estimates: each regime selected among the same frozen C01–C20 bank only after all 20 checkpoints were frozen.

## ST post-selection context diagnostics

Frozen C07 was evaluated without retraining on the 1,072 context-only ST rows:

- unseen-source earlier ENV-03 (609): F1 **0.7056**, precision 0.9679, recall 0.5551;
- known-source future ENV-01/02/04 (463): F1 **0.7649**, precision 0.6566, recall 0.9160;
- post-hoc combined ENV-03 view (770): F1 **0.7052**.

These context results are descriptive and cannot alter Best ST.

## Interpretation guardrails

O is highly source-joint and near-frame-correlated, while S, T and ST answer stricter and different generalization questions. Raw F1 values across regimes are therefore not a global model ranking.

ST is the strictest primary regime. C07 has F1 0.7040 with frame-bootstrap 95% CI approximately **0.609–0.790** and video-cluster CI approximately **0.465–0.859**.

## Evidence

See:
- `reports/scratch/O_REPORT.md`
- `reports/scratch/S_REPORT.md`
- `reports/scratch/T_REPORT.md`
- `reports/scratch/ST_REPORT.md`
- `manifests/scratch/O_REGISTRY.yaml`
- `manifests/scratch/S_REGISTRY.yaml`
- `manifests/scratch/T_REGISTRY.yaml`
- `manifests/scratch/ST_REGISTRY.yaml`

## Frozen scratch rules

- same C01–C20 bank in O/S/T/ST;
- exactly 20 epochs;
- no validation or early stopping;
- no primary augmentation;
- final epoch-20 checkpoint;
- threshold 0.5;
- all candidates frozen before each regime test opened;
- champion order: F1 → balanced accuracy → PR-AUC → lower parameter count → candidate ID.

## Next gate

**Phase 9 — cross-regime synthesis and uncertainty:** produce the champion envelope and fixed-C14 robustness view with 5,000 frame bootstrap and 5,000 video/group bootstrap resamples using a fixed seed.
