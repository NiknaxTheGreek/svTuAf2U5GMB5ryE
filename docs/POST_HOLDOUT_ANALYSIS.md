# Post-Holdout Error and Robustness Analysis

## Status

Step 8 post-holdout diagnostics: COMPLETE.

This stage is strictly analysis-only. The reported final test metrics, selected checkpoint, threshold, and model remain unchanged.

## Governance

The Step 7 holdout has already been consumed. Therefore Step 8 does not:

- retrain the model,
- select a different epoch,
- tune the threshold,
- change preprocessing,
- replace the reported final test result.

All results below are diagnostic observations on the already-consumed holdout.

## Source-group uncertainty

A stratified source-group bootstrap with 20,000 replicates was run by independently resampling the 13 flip and 11 notflip test source groups.

Approximate 95% bootstrap intervals:

- F1: 0.6142 to 0.8829
- Precision: 0.6920 to 0.9888
- Recall: 0.5048 to 0.8558
- Accuracy: 0.6601 to 0.8996

The wide intervals are consistent with strong between-clip variation and the relatively small number of independent source groups.

## Confidence behavior

Prediction confidence was measured as absolute distance from the 0.5 decision threshold.

- Mean confidence margin for correct frames: 0.3898
- Mean confidence margin for errors: 0.2612

Error rate by confidence margin:

- 0.00-0.05: 60.0%
- 0.05-0.10: 35.5%
- 0.10-0.20: 46.6%
- 0.20-0.30: 32.9%
- 0.30-0.40: 30.2%
- 0.40-0.50: 8.8%

This shows that low-margin predictions are substantially less reliable, although confidence is not perfectly monotonic with error rate.

## Sequence behavior

Across the 24 test source groups:

- 11 groups contain zero errors.
- 13 groups contain at least one error.
- Median number of prediction-state transitions per group: 1.
- Maximum transitions in a group: 8.
- Longest consecutive error run: 15 frames.

The worst source groups are:

- `flip__0041`: 18 / 20 incorrect, mean predicted flip probability 0.199
- `notflip__0053`: 23 / 30 incorrect, mean predicted flip probability 0.623
- `flip__0038`: 32 / 46 incorrect, mean predicted flip probability 0.395
- `flip__0042`: 16 / 30 incorrect, mean predicted flip probability 0.425
- `flip__0047`: 11 / 25 incorrect, mean predicted flip probability 0.527

Several of these are persistent clip-level failures rather than isolated frame mistakes.

## Frame-position analysis

Error rates across relative frame position within each source group:

All classes:

- Q1: 23.8%
- Q2: 12.8%
- Q3: 20.9%
- Q4: 23.7%
- Q5: 26.4%

Flip clips:

- Q1: 37.5%
- Q2: 17.9%
- Q3: 33.3%
- Q4: 40.4%
- Q5: 33.9%

Notflip clips:

- Q1: 9.7%
- Q2: 8.2%
- Q3: 9.8%
- Q4: 8.2%
- Q5: 19.4%

The strongest position-dependent variation occurs for flip clips. This is compatible with the idea that some temporal phases of a page-turning event are visually harder to classify from a single frame.

## Fixed perturbation robustness battery

The frozen model and threshold were evaluated under fixed mild image perturbations.

Original final-test F1: 0.75145

- Brightness +15%: F1 0.69636, delta -0.05509
- Contrast +15%: F1 0.70281, delta -0.04863
- JPEG quality 70: F1 0.74319, delta -0.00825
- Gaussian blur radius 1: F1 0.74903, delta -0.00241
- Contrast -15%: F1 0.75049, delta -0.00095
- Brightness -15%: F1 0.75836, delta +0.00692

The model is therefore much more sensitive to moderate increases in brightness or contrast than to mild blur or JPEG recompression. The slight F1 increase under darker images is diagnostic only and must not be used to change preprocessing after test access.

## Interpretation

The dominant limitation is not random frame noise. It is source-group/domain variation:

1. A subset of clips fails persistently across many consecutive frames.
2. Flip recall varies strongly by clip and frame position.
3. Low-confidence predictions are substantially less reliable.
4. Brighter / higher-contrast shifts reduce recall and F1 materially.
5. Mild blur and JPEG compression have comparatively little effect.

These findings support the next planned extension: sequence-level aggregation of the existing frame probabilities, evaluated only as a post-holdout downstream analysis rather than as a replacement for the already-reported single-image final result.
