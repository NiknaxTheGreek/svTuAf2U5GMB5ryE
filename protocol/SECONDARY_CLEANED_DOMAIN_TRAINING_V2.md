# Phase 11B — Cleaned-domain training and optimization ladder

Status: **user-approved secondary/exploratory extension**.

This procedure is separate from the frozen V2 primary champion selection. It cannot redefine Best O/S/T/ST.

## Domains

Primary mandatory transformed domain:

- **hand/arm-cleaned** images produced by the frozen inpainting pipeline after its context-only QA gates pass.

The same training ladder may later be applied to the **naturalistic chroma counterfactual** after its context-only QA gate passes. Grayscale and matched-chroma rotation remain primarily perturbation controls unless explicitly promoted to transformed training domains before their transformed tests are opened.

## Regime-specific populations

Preserve the exact frozen O/S/T/ST row memberships. Transformation changes pixels only; it never changes labels, sample IDs, video IDs, frame numbers, or regime membership.

| Regime | Original frozen champion | Train/test membership |
|---|---|---|
| O | C14 | exact O train / O test |
| S | C18 | exact S train / S test |
| T | C06 | exact T train / T test |
| ST | C07 | exact ST train / ST primary test |

## Four-step hand-cleaned ladder for every regime

### A. Original benchmark

Use the already-frozen original champion checkpoint on the original test pixels.

No new computation is needed except metric reconciliation from saved predictions.

### B. Counterfactual inference

Use the **same frozen original champion checkpoint** on the hand/arm-cleaned version of that regime's test images.

No retraining, no threshold change, no test-driven transform change.

This asks whether the originally learned model is sensitive to removal of hand/arm information.

### C. Fixed-config cleaned retrain

Initialize a fresh scratch CNN with the **exact hyperparameters of that regime's frozen original champion**:

- O -> C14 hyperparameters;
- S -> C18;
- T -> C06;
- ST -> C07.

Train from scratch on the hand/arm-cleaned version of the exact regime training partition.

Training policy remains:

- 20 epochs;
- no early stopping;
- threshold 0.5;
- same optimizer and all candidate hyperparameters;
- same seed policy;
- final epoch-20 checkpoint.

Evaluate once on the matching hand/arm-cleaned test partition.

This asks whether the same architecture/configuration can learn the task when hands/arms are absent from both learning and deployment.

### D. Cleaned-domain hyperparameter optimization

Perform HPO **inside the cleaned training population only**. The cleaned regime test remains sealed until a cleaned-domain configuration is frozen.

For comparability, use the same ScratchCNN model family and the same V2 hyperparameter search space. Use a fixed non-adaptive candidate budget and a prospectively frozen candidate set/search seed. Do not use Bayesian/adaptive search after observing any cleaned test metric.

Inner model selection must mirror the scientific isolation of the outer regime as closely as practical:

- O: video-grouped inner split so adjacent frames from the same canonical video do not cross inner train/validation;
- S: source/video-grouped inner split within the S training population;
- T: nested chronological inner split within the T training population;
- ST: source-aware plus chronological inner split within the ST training population.

The inner split and HPO candidate set must be frozen before any cleaned test evaluation.

After HPO:

1. freeze the selected cleaned-domain configuration;
2. train it on the full cleaned regime training partition for the fixed 20 epochs;
3. evaluate exactly once on the cleaned regime test.

The optimized cleaned-domain model remains secondary and does not replace the original V2 champion.

## Required comparisons

For each O/S/T/ST regime report:

1. original champion -> original test;
2. original champion -> cleaned test;
3. original champion hyperparameters retrained on cleaned train -> cleaned test;
4. cleaned-domain HPO winner trained on cleaned train -> cleaned test.

Report F1, precision, recall, accuracy, balanced accuracy, ROC-AUC, PR-AUC and confusion matrix.

For paired evaluations on identical test samples, report 5,000 frame and 5,000 video/group bootstrap confidence intervals for F1 differences where meaningful.

## Interpretation examples

- B drops, C recovers: the task remains learnable without hands/arms, but the original learned representation was sensitive to the original hand-containing distribution.
- B and C both remain poor: either the removed region carried task-relevant information or the inpainting transformation damaged other useful evidence; inspect QA before causal interpretation.
- C is strong and D adds little: the original champion hyperparameters transfer well to the cleaned domain.
- D materially exceeds C: the cleaned domain appears to favor a different configuration, but this is a secondary transformed-domain result only.

No transformed-domain result may retroactively redefine Best O/S/T/ST.
