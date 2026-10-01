# Secondary perturbation development protocol V2

Status: **development only; primary O/S/T/ST champions remain frozen and authoritative**.

## Purpose

Develop and freeze two image perturbations before any champion test-set ablation is run:

1. colour removal (grayscale);
2. hand/arm masking.

This phase measures perturbation quality only. It does **not** use classifier predictions, F1, accuracy, ROC-AUC, PR-AUC, threshold changes, retraining, or champion selection.

## Development population

Use only a fixed stratified sample from the 1,072 ST `context` rows. No ST primary-test row is allowed.

The sample is deterministic: six images from each `(role_detail, label, environment_id)` context stratum, ordered by SHA-256 of a fixed development seed plus `sample_id`. This gives 48 development images spanning:

- held-out ENV-03 earlier-frame context;
- ENV-01/02/04 later-frame context;
- both flip and not-flip classes.

## Grayscale candidates

Compare:

- prospectively defined Rec.709 luma replicated to R/G/B;
- Pillow `L` conversion replicated to R/G/B as a reference.

Selection criterion is luminance preservation, not model performance. The preferred transform should remove chroma while minimally changing the prospectively defined luminance signal.

## Hand/arm masking candidates

Evaluate deterministic skin-region candidates built from YCbCr thresholds, an optional normalized-RGB gate, morphology, connected-component cleanup, and an optional image-boundary prior. Candidate masks are overlaid on development images for visual review.

The final hand-mask algorithm is frozen only after visual review confirms that it removes visible hand/arm regions with acceptable page/background collateral masking across the fixed development sample.

Full-scale hand-removal remains gated by this proof of concept, consistent with the project action plan.

## Replacement and control

When the mask is eventually applied:

- masked pixels are filled with the median colour of a local unmasked border ring;
- a matched-control perturbation uses the same mask area mirrored horizontally and the same fill method;
- this helps distinguish a hand-specific effect from generic occlusion damage.

## Freeze rule

After visual review, commit one exact grayscale transform and one exact hand-mask configuration. Their code/configuration hashes are frozen before applying them to any O/S/T/ST primary test population.

Any later evaluation on an already-opened primary test remains explicitly **post-hoc/exploratory** and cannot modify Best O/S/T/ST.
