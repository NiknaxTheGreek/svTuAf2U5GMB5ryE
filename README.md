# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Verified scratch-CNN evidence

| Regime | Champion | F1 | Frame 95% CI | Video/group 95% CI | Interpretation |
|---|---:|---:|---:|---:|---|
| O | C14 | **0.9983** | 0.995–1.000 | 0.994–1.000 | supplied source-joint benchmark |
| S | C18 | **0.8196** | 0.788–0.850 | 0.674–0.915 | held-out ENV-03 |
| T | C06 | **0.8800** | 0.852–0.907 | 0.837–0.917 | later frames of known videos |
| ST | C07 | **0.7040** | 0.606–0.790 | 0.466–0.857 | held-out ENV-03 + later frames |

All intervals use 5,000 resamples with fixed seed 20261001 and are selection-conditioned.

## Fixed O-winning configuration

C14, retrained independently inside each regime with the same architecture/hyperparameters:

| Regime | F1 |
|---|---:|
| O | 0.9983 |
| S | 0.6936 |
| T | 0.2981 |
| ST | 0.4646 |

The T result retains high ROC-AUC/PR-AUC despite low fixed-threshold F1, consistent with substantial operating-threshold/calibration shift.

## ST context diagnostics

Frozen C07, without retraining:

- unseen-source earlier ENV-03: F1 **0.7056**;
- known-source future ENV-01/02/04: F1 **0.7649**;
- combined ENV-03 post-hoc descriptive view: F1 **0.7052**.

These context results cannot alter Best ST.

## Interpretation guardrail

O, S, T and ST answer different generalization questions. Their raw F1 values are not a global model ranking.

See `reports/synthesis/CROSS_REGIME_SYNTHESIS.md` and `manifests/synthesis/PHASE9_REGISTRY.yaml`.

## Next gate

Freeze and execute the **single fixed ImageNet ResNet18 comparator recipe**. It is separate from scratch champion selection: no sweep, no validation, no early stopping, final epoch-20 checkpoint, threshold 0.5.
