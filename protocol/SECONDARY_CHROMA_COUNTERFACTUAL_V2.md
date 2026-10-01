# Phase 11B — Expanded chroma counterfactual analysis

Status: **prospectively frozen secondary/exploratory analysis**.

The existing Rec.709 grayscale result is retained as the crude chroma-removal ablation. It is not by itself sufficient to distinguish useful colour dependence from generic grayscale distribution shift.

## Variants

### 1. Original RGB
Reference.

### 2. Rec.709 grayscale
Already evaluated. It preserves the specified luminance signal while removing chroma.

### 3. Matched chroma perturbation
Convert to YCbCr, preserve Y, center chroma around neutral (128,128), and rotate the chroma vector by **+90 degrees**.

This preserves chroma magnitude before unavoidable RGB-gamut clipping while strongly changing hue. It is the colour analogue of a matched perturbation control.

### 4. Naturalistic colour counterfactual
Keep the target image's luminance and geometry. Replace only its chroma statistics using a deterministic context-only donor palette through a frozen mean/covariance colour-transfer transform.

The donor comes from a different environment where possible but the same class label. Donor selection is deterministic and independent of classifier behaviour.

## Development/quality gate

Use the same fixed 100-image ST-context PoC population as the inpainting experiment.

No primary test prediction is used to tune either chroma transform.

Before O-test evaluation:

- verify luminance preservation;
- record clipping rate;
- visually review all naturalistic counterfactuals for plausible document colour;
- reject the naturalistic method entirely if it introduces substantial geometry/text artifacts.

## Evaluation

If the quality gate passes, frozen Best-O C14 is evaluated on identical O-test images for all four variants.

Use threshold 0.5 plus 5,000 paired frame and 5,000 paired video/group bootstrap resamples.

Interpretation is hierarchical:

- grayscale alone fails -> possible grayscale-domain-shift sensitivity;
- grayscale + matched chroma control fail -> broader colour-distribution sensitivity;
- naturalistic counterfactual also fails -> stronger evidence that chromatic cues themselves matter.

No result can replace Best O.
