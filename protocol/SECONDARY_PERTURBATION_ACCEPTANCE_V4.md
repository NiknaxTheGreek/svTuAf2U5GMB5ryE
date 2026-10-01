# Secondary perturbation acceptance gates V4

Status: **development only**. No O/S/T/ST primary-test perturbation evaluation may begin until these gates pass.

## Scientific target

The hand perturbation must remove the entire visible hand/arm signal while preserving every non-hand pixel exactly.

The colour perturbation must remove chromatic information while preserving luminance, geometry, edge structure, resolution, and spatial alignment as closely as numerically possible.

No classifier prediction, label-conditioned tuning, F1, accuracy, ROC-AUC, PR-AUC, or champion ranking may be used to choose or tune either perturbation.

## Hand/arm removal hard gates

1. **Visible-hand recall:** 100% of development/QA images containing a visible hand or arm must have all visible hand/arm regions covered by the final mask.
2. **Residual-hand gate:** visual QA must find zero visible residual hand/arm regions after masking.
3. **Outside-mask invariance:** all pixels outside the final mask must be bit-for-bit identical to the original preprocessed model input.
4. **Collateral-loss target:** minimize non-hand pixels inside the mask. Page text, page edges, diagrams, and book geometry must not be deliberately removed merely to increase hand recall.
5. **No hallucinated reconstruction:** do not use generative inpainting to claim preservation of information hidden by the hand. Pixels physically occluded by the hand cannot be recovered from the source image.
6. **Blind correction rule:** if automatic segmentation cannot satisfy the 100% removal gate, manual mask correction is allowed only as a predefined quality-control stage, performed without viewing classifier predictions or using model performance to guide the correction.
7. **Auditability:** preserve the automatic mask, final QA-corrected mask if applicable, changed-pixel count, unchanged-pixel count, and mask hash for every image.

A fully automatic method may be reported separately from the QA-corrected method; only a method that passes the 100% visible-removal gate may be called complete hand removal.

## Colour-removal hard gates

1. Remove chroma completely: R=G=B for every transformed pixel.
2. Perform the transform at floating-point model input where possible to avoid an unnecessary second uint8 quantization.
3. Preserve linear-light sRGB/Rec.709 luminance to numerical precision.
4. Preserve dimensions, spatial coordinates, padding, and all image geometry exactly.
5. Do not blur, denoise, resize, sharpen, crop, or otherwise alter structure.
6. Report maximum and mean luminance error, chroma residual, and changed-pixel accounting.

## Development and QA populations

Algorithm parameters are developed only from ST context rows. The 161 ST primary-test images remain excluded from algorithm development.

Before primary-test use, perform a second context-only QA pass that was not used for parameter selection. The preferred design is:

- fixed tuning subset;
- separate fixed QA subset;
- freeze;
- only then primary-test application.

## Primary-test application

Once frozen, the exact same transformation code and parameters are applied unchanged to the ST primary-test set. Any human QA/correction procedure must also be frozen prospectively and must remain blind to classifier predictions.

The resulting experiment remains post-hoc/exploratory and cannot alter Best O/S/T/ST.
