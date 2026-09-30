# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`. The V2 rebuild is designed to separate conventional benchmark performance from source and temporal generalization.

## Current verified state

**Phase 1 — raw-data audit: complete. Model training has not started.**

- Official archive identity verified: 939,921,132 bytes.
- SHA-256: `033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f`.
- 2,989 / 2,989 images accounted for; zero decode failures.
- Supplied split: 2,392 training / 597 testing.
- 117 canonical videos.
- No exact decoded-pixel duplicate groups.
- All 597 supplied-test images come from videos represented in supplied training.
- 573/597 supplied-test images are only one frame number from a training frame from the same video.
- Four reviewed source/environment groups cover all 117 videos exactly once.

See `reports/data_audit/DATA_AUDIT.md` for the evidence and interpretation.

## Frozen primary scratch-CNN rules

- Python / PyTorch.
- Four regimes: Original (O), Source-Safe (S), Temporal (T), Source-Safe + Temporal (ST).
- Same frozen 20-candidate bank in every regime.
- Exactly 20 training epochs per candidate.
- No validation split for primary scratch-CNN selection.
- No validation-driven early stopping.
- No Bayesian/adaptive search.
- Final epoch-20 checkpoint.
- Threshold = 0.5.
- All candidates in a regime are trained and frozen before that regime test is evaluated.
- Champion tie order: F1 → balanced accuracy → PR-AUC → lower trainable parameter count → candidate ID.
- Because a regime test selects among frozen candidates, its result is reported as a **selection-benchmark estimate**.
- Every supplied image must have an explicit role in every regime.

## Next gate

Construct, audit and freeze the O/S/T/ST membership manifests with complete 2,989-image accounting **before any model training**.
