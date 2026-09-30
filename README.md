# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild separates conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 2 — dataset and O/S/T/ST split audit: complete. Model training has not started.**

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

Temporal rule: sort each canonical video by `FrameNumber`, then use `floor(0.80 × n)` observed images as the earlier portion.

Source-safe holdout: **ENV-03**.

See:
- `reports/data_audit/DATA_AUDIT.md`
- `reports/splits/SPLIT_AUDIT.md`
- `manifests/splits/SPLIT_REGISTRY.yaml`

## Frozen scratch-CNN rules

- Python / PyTorch.
- Same frozen 20-candidate bank in O, S, T and ST.
- Exactly 20 epochs per candidate.
- No validation split for primary candidate selection.
- No validation-driven early stopping.
- No Bayesian/adaptive search.
- Final epoch-20 checkpoint.
- Threshold = 0.5.
- All 20 candidates in a regime must be trained and frozen before that regime test is evaluated.
- Champion tie order: F1 → balanced accuracy → PR-AUC → lower trainable parameter count → candidate ID.
- Because each regime test selects among frozen candidates, results are reported as **selection-benchmark estimates**.

## Next gate

Audit the frozen 20-candidate bank formally and build leakage-safe majority and handcrafted-feature baselines before scratch-CNN training.
