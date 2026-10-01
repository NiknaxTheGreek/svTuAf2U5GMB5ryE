# Phase 11B — Hand/arm inpainting counterfactual

Status: **prospectively frozen secondary/exploratory PoC**.

The earlier hand-region experiment remains valid as an **occlusion sensitivity** analysis. It is not treated as realistic hand removal.

## Recovered historical method

Before inventing a new hand-removal pipeline, the Project 4 history was checked directly.

Historical commit:

`0b06f7163998c06135c6ea18d286f02fcedb9caa`

Recovered script:

`scripts/hand_arm_clean_dataset.py`

The exact historical algorithm is:

1. **SCHP Pascal-7 human parsing** from `pirocheto/schp-pascal-7` for upper/lower arm labels.
2. **MediaPipe Hands**, up to two hands, minimum detection confidence 0.35.
3. Convex-hull hand masks plus scale-relative dilation.
4. Union of SCHP arm mask and MediaPipe hand mask.
5. Conservative morphological closing and dilation around the combined region.
6. **OpenCV TELEA inpainting**, radius 5, performed at half resolution.
7. Cubic resize back to the original resolution.
8. **Gaussian-feathered compositing** so unchanged pixels remain original and the reconstructed region blends into the image.

The historical source is preserved unchanged in the V2 branch. Only V2 data/input compatibility code may surround it.

## Historical full-corpus corroboration

A different historical implementation, `scripts/build_hand_arm_removed_dataset.py`, ran successfully on all **2,989** images in workflow run **36145174745**.

That implementation used LRASPP MobileNetV3-Large person segmentation, threshold 0.25, connected-component cleanup, 7×7 dilation and TELEA radius 5.

Its preserved artifacts remain live until 24 December 2026:

- cleaned dataset artifact: 811,837,610 bytes, artifact 10870977276;
- QA artifact: 10870663479.

This is useful historical evidence that a full-corpus cleaning workflow was operational. It is not the preferred V2 hand/arm counterfactual because it segments generic person pixels rather than explicitly separating arms and hands.

## Stage 1 — exact resurrection on the existing 48-image context sample

Use the same fixed 48 **ST context-only** development images already used for perturbation QA.

No primary test image is allowed.

Run the recovered SCHP + MediaPipe + TELEA pipeline **without changing its algorithm**.

Generate side-by-side before/after previews for all 48 images and record:

- total mask fraction;
- SCHP arm pixels;
- MediaPipe hand pixels;
- zero-mask cases;
- failures/exceptions.

### Stage-1 visual gate

Inspect all 48 images for:

- complete enough hand removal;
- wrist/forearm/arm coverage;
- plausible reconstructed page/background;
- preservation of page geometry;
- preservation of unaffected text/background;
- visible seams or unrealistic flat regions.

Classifier predictions are prohibited during this gate.

## Improvement rule

Do **not** invent a new V3/V4 pipeline.

If the exact historical method reveals a concrete defect, only one minimal targeted correction may be introduced at a time, with the reason documented before re-running QA.

Classifier performance may never determine which correction is chosen.

## Stage 2 — 100-image context-only validation

If Stage 1 is acceptable, apply the same frozen pipeline to the already-frozen deterministic 100-image ST-context PoC set.

Quality gate:

- at least 90% adequate visible hand/arm removal;
- at most 10% major visible inpainting artifact;
- document geometry preserved;
- text/background outside the removal region preserved;
- outside-mask changes negligible except the feather/seam ring.

If the gate fails, stop and report the failed PoC.

## Full-scale evaluation only if both gates pass

Freeze the automatic pipeline, then evaluate frozen Best-O C14 on identical O-test images:

- original;
- realistic inpainted no-hand/arm counterfactual.

Retain the earlier hand-mask and matched-occlusion analyses as controls.

Use threshold 0.5 and 5,000 paired frame plus 5,000 paired video/group bootstrap resamples.

This remains post-hoc and cannot change Best O.
