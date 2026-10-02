# Phase 11B — Expanded chroma counterfactual analysis

Status: **prospectively frozen secondary/exploratory analysis**.

This analysis is post-hoc and cannot replace Best O/S/T/ST.

## Scientific question

Does the frozen model depend specifically on chromatic information, or does the previously observed grayscale collapse mostly reflect an unnatural distribution shift?

## Frozen champions

The eventual test evaluation is performed independently for all four frozen scratch champions:

- O -> C14;
- S -> C18;
- T -> C06;
- ST -> C07.

No retraining and no threshold changes occur in the counterfactual sensitivity experiment.

## Variants

### 1. Original RGB

Reference.

### 2. Rec.709 grayscale

Replicate

`Y = 0.2126 R + 0.7152 G + 0.0722 B`

into all three RGB channels.

This is the existing crude chroma-removal ablation.

### 3. Matched chroma rotation

Represent each pixel as Rec.709 luminance Y plus two YCbCr-style chroma coordinates.

Keep Y fixed and rotate the centered chroma vector by exactly **+90 degrees**.

Instead of clipping invalid RGB values, radially compress the rotated chroma vector toward neutral until the reconstructed RGB value is in gamut. This preserves luminance and hue-rotation direction while avoiding brightness changes caused by clipping.

This is the matched colour-distribution control.

### 4. Naturalistic palette counterfactual

Keep the target image's Rec.709 luminance and all spatial geometry fixed.

Use a deterministic donor image only to define global chroma mean/covariance statistics. Transform the target's two-dimensional chroma distribution toward the donor distribution through a regularized whitening/coloring transform at fixed strength **0.75**.

The donor's spatial pixels are never pasted into the target.

Per-pixel radial gamut compression preserves Y while ensuring valid RGB.

## Donor independence

The target's true label is **never used** to choose a donor.

The donor pool is the intersection of images that are **non-test in O, S, T and ST simultaneously**.

For each target:

1. exclude itself;
2. prefer a different `environment_id` and different canonical video;
3. deterministically order eligible donors by SHA256 of `target_sample_id | donor_sample_id`;
4. use the first eligible donor.

No classifier prediction is involved.

## Context-only quality gate

Use the already-frozen deterministic **100-image ST-context** PoC sample.

No primary test image and no classifier output may be used for transform development or acceptance.

Review all 100 images.

Pre-frozen quantitative requirements:

- matched rotation mean Rec.709 luminance MAE <= 0.5;
- naturalistic transfer mean Rec.709 luminance MAE <= 0.5;
- 95th percentile of per-image naturalistic luminance MAE <= 1.0;
- median naturalistic chroma displacement >= 3.0, so the counterfactual is non-trivial;
- no geometry/text-layout modification by construction.

Visual requirement:

- no more than 10% of naturalistic counterfactuals may show a major implausible colour artifact.

If the naturalistic variant fails, it is rejected rather than tuned on primary test images.

## Frozen-model evaluation if the gate passes

For each regime, evaluate the same frozen checkpoint on its exact test membership in four pixel variants:

1. original RGB;
2. Rec.709 grayscale;
3. matched chroma rotation;
4. naturalistic palette counterfactual.

Report F1, precision, recall, accuracy, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix, probability changes and changed predictions.

Use 5,000 paired frame bootstrap and 5,000 paired video/group bootstrap resamples for F1 differences against original RGB.

## Interpretation hierarchy

- only grayscale drops -> consistent with grayscale/domain-shift sensitivity;
- grayscale and matched rotation drop similarly -> broader colour-distribution sensitivity;
- a material drop under the naturalistic counterfactual -> stronger evidence that chromatic cues themselves influence the frozen model.

No chroma result can redefine Best O/S/T/ST.
