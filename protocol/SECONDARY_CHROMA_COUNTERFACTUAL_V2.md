# Phase 11C — Expanded chroma counterfactual analysis

Status: **prospectively frozen secondary/exploratory analysis**.

The existing Rec.709 grayscale result is retained as the crude chroma-removal ablation. It cannot by itself distinguish dependence on useful chromatic information from sensitivity to an unnatural grayscale domain shift.

## Frozen models

The eventual sensitivity analysis applies to all four already-frozen scratch champions:

- O: C14;
- S: C18;
- T: C06;
- ST: C07.

Each model is evaluated only on its own frozen test membership. No result can replace Best O/S/T/ST.

## Variants

### 1. Original RGB

Reference.

### 2. Rec.709 grayscale

Replicate Rec.709 luminance into all RGB channels. This is the existing destructive chroma-removal ablation.

### 3. Matched chroma rotation

Convert RGB to full-range BT.709 YCbCr.

Preserve target Y and rotate the centered chroma vector `(Cb-0.5, Cr-0.5)` by exactly **+90 degrees**.

The vector magnitude is therefore unchanged before RGB gamut clipping, while hue is strongly altered. RGB reconstruction uses iterative luminance correction after clipping to keep the final image as close as possible to the original target luminance.

This is the colour analogue of a matched perturbation control.

### 4. Naturalistic palette counterfactual

Keep target BT.709 luminance and all target spatial structure.

Transfer only the target CbCr distribution using a frozen whitening/colouring transform so that the transformed chroma has the deterministic donor image's mean and covariance.

Donor selection is:

- ST context-only;
- deterministic;
- from a different environment where possible;
- **independent of the target's true label**;
- independent of all classifier predictions.

No geometry, crop, warp, blur or spatial resampling is allowed.

## Development population

Use the frozen deterministic 100-image ST-context PoC sample.

No primary O/S/T/ST test image and no classifier prediction may participate in transform development or QA.

## Quantitative QA gates

For both colour transforms:

- mean absolute Rec.709 luminance error <= 1.0 intensity level;
- 95th-percentile absolute luminance error <= 3.0 levels.

Matched rotation additionally requires RGB gamut clipping on no more than 20% of channel values.

Naturalistic transfer requires clipping on no more than 15% of channel values.

All 100 images are visually reviewed. At least 90% of naturalistic counterfactuals must look like plausible RGB document/environment images without obvious false-colour artifacts.

## Frozen-model evaluation if QA passes

For **each** O/S/T/ST champion, evaluate the exact same test sample IDs under:

1. original RGB;
2. Rec.709 grayscale;
3. matched chroma rotation;
4. naturalistic colour counterfactual.

No retraining and no threshold change occur in this sensitivity stage.

Use threshold 0.5 and 5,000 paired frame plus 5,000 paired video/group bootstrap resamples.

## Interpretation

- collapse only under grayscale: consistent with grayscale-domain-shift sensitivity;
- similar degradation under grayscale and matched chroma rotation: broader sensitivity to the colour distribution;
- material degradation under a visually plausible naturalistic palette counterfactual: stronger evidence that chromatic relationships genuinely contribute to the learned decision rule.

## Transformed-domain extension

If the naturalistic counterfactual passes context-only QA, it may also be used as a transformed training domain:

- retrain O/S/T/ST using each original champion's fixed hyperparameters on transformed train and transformed test;
- optionally perform separately frozen transformed-domain HPO using transformed training data only;
- keep transformed test sealed until the HPO configuration is frozen.

These remain secondary experiments and cannot redefine the V2 primary champions.
