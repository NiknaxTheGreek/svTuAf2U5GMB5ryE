# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 3 — data, splits, frozen candidate bank and non-CNN baselines: complete. Scratch-CNN training has not started.**

Dataset:
- 2,989 / 2,989 images accounted for.
- Official archive SHA-256: `033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f`.
- 117 canonical videos.
- zero exact decoded-pixel duplicate groups.
- all 597 supplied-test images come from videos represented in supplied training.

Frozen V2 regimes:

| Regime | Train | Primary test | Context | Excluded |
|---|---:|---:|---:|---:|
| O | 2,392 | 597 | 0 | 0 |
| S | 2,219 | 770 | 0 | 0 |
| T | 2,365 | 624 | 0 | 0 |
| ST | 1,756 | 161 | 1,072 | 0 |

Temporal rule: sort each canonical video by `FrameNumber`, then use `floor(0.80 × n)` observed images as the earlier portion. Source-safe holdout: **ENV-03**.

## Frozen candidate bank

The canonical 20-config scratch-CNN bank passed its formal pre-training audit:

- SHA-256: `a4bebfc6c21df01e9294dacccb26ba84602cb3800b20414ad1d51f6f343c3b12`;
- 20 candidate IDs, C01–C20;
- 20 unique hyperparameter configurations;
- all values inside the frozen search space;
- fixed 20 epochs, no validation, no early stopping, final epoch-20 checkpoint, threshold 0.5.

Scratch-CNN candidate test results remain unopened.

## Leakage-safe baselines

The majority and handcrafted logistic-regression protocols were frozen before baseline tests were evaluated. All feature sets A/B/C are reported; no feature family was selected post-hoc.

| Regime | Majority F1 | LogReg A F1 | LogReg B F1 | LogReg C F1 |
|---|---:|---:|---:|---:|
| O | 0.0000 | 0.7842 | 0.8571 | 0.9268 |
| S | 0.0000 | 0.4568 | 0.6583 | 0.5560 |
| T | 0.0000 | 0.7854 | 0.7450 | 0.8043 |
| ST | 0.0000 | 0.3656 | 0.5714 | 0.1687 |

The complete baseline execution reproduced identical core result hashes three times. Training-only `StandardScaler` parameters, coefficients and per-sample predictions are retained.

These results show that simple visual statistics contain substantial signal, especially under O, but that the same feature representations do not transfer uniformly under source shift. In S and ST, some feature sets retain high ROC/PR ranking while producing low fixed-threshold recall, which is consistent with distribution/calibration shift. The threshold remains fixed at 0.5.

See:
- `reports/data_audit/DATA_AUDIT.md`
- `reports/splits/SPLIT_AUDIT.md`
- `reports/baselines/BASELINE_REPORT.md`
- `manifests/baselines/BASELINE_REGISTRY.yaml`

## Frozen scratch-CNN rules

- Python / PyTorch.
- Same frozen 20-candidate bank in O, S, T and ST.
- Exactly 20 epochs per candidate.
- No validation split for primary candidate selection.
- No validation-driven early stopping.
- No Bayesian/adaptive search.
- Final epoch-20 checkpoint.
- Threshold = 0.5.
- **All 20 candidates in a regime must be trained and frozen before that regime test is evaluated.**
- Champion tie order: F1 → balanced accuracy → PR-AUC → lower trainable parameter count → candidate ID.
- Because each regime test selects among frozen candidates, results are reported as **selection-benchmark estimates**.

## Next gate

Freeze the exact scratch-CNN implementation and training/evaluation machinery. Then train **all 20 O candidates without consulting the O test**, verify all final checkpoints, and only afterward open the O test for a single batch evaluation.
