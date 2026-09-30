# MonReader — V2 scientific rebuild

This repository is being rebuilt from a clean methodological baseline for Apziva Project 4 (MonReader: `flip` vs `notflip`).

Current authoritative protocol: **train/test-only V2**.

The active project must not use the historical validation/Bayesian/source-gate workflow. Pre-reset work is preserved under dated historical Git branches and is non-authoritative unless reproduced under V2.

## Frozen primary rules

- Python / PyTorch.
- Four scratch-CNN regimes: Original (O), Source-Safe (S), Temporal (T), Source-Safe + Temporal (ST).
- The same frozen 20-candidate bank is used in every regime.
- Exactly 20 training epochs per candidate.
- No validation split in primary scratch selection.
- No validation-driven early stopping.
- No Bayesian/adaptive search.
- Final epoch-20 checkpoint.
- Classification threshold = 0.5.
- All 20 candidates for a regime must be trained and frozen before that regime test is evaluated.
- Champion selection: F1 → balanced accuracy → PR-AUC → lower trainable parameter count → candidate ID.
- Because the test selects among frozen candidates, results are described as **selection-benchmark estimates**.
- Every supplied image must receive an explicit role in every experimental regime.

No model training has been performed under V2 yet.

The next execution gate is a complete raw-data identity and integrity audit.
