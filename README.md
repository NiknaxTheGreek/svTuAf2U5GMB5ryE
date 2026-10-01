# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`.

## Frozen scratch-CNN champions

| Regime | Champion | F1 | Interpretation |
|---|---:|---:|---|
| O | C14 | **0.9983** | supplied source-joint benchmark |
| S | C18 | **0.8196** | held-out ENV-03 |
| T | C06 | **0.8800** | later frames of known videos |
| ST | C07 | **0.7040** | held-out ENV-03 + later frames |

See `reports/synthesis/CROSS_REGIME_SYNTHESIS.md` for 5,000-resample frame and video/group uncertainty.

## Fixed ImageNet ResNet18 comparator

One prospectively frozen non-adaptive ResNet18 recipe was trained independently in O/S/T/ST: 5 head-only epochs + 15 full-network epochs, AdamW, final epoch-20 checkpoint, no validation, no early stopping, threshold 0.5.

| Regime | Scratch champion F1 | ResNet18 F1 | ΔF1 (ResNet − scratch) | Paired video/group 95% CI |
|---|---:|---:|---:|---:|
| O | 0.9983 | 0.9931 | -0.0052 | -0.014 to +0.002 |
| S | 0.8196 | 0.7807 | -0.0390 | -0.169 to +0.076 |
| T | 0.8800 | 0.8807 | +0.0007 | -0.032 to +0.030 |
| ST | 0.7040 | 0.7196 | +0.0156 | -0.212 to +0.259 |

None of the four paired video/group intervals excludes zero. The fixed pretrained comparator therefore does not support replacing any frozen scratch champion.

See `reports/resnet18/RESNET18_COMPARATOR_REPORT.md` and `manifests/resnet18/RESNET18_REGISTRY.yaml`.

## Key interpretation

O, S, T and ST answer different generalization questions and are not a global model ranking. O is highly source-joint and near-frame-correlated; S isolates unseen-source difficulty; T isolates later-frame difficulty within known sources; ST combines unseen source and later-frame stress.

## Next gate

**Phase 11 — controlled secondary analyses.** These are explicitly post-selection and cannot change Best O/S/T/ST. Priority analyses include shortcut/hand-arm dependence, controlled augmentation, visual/data ablations, and temporal/video deployment diagnostics.
