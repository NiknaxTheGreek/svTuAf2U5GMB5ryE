# Phase 11B — Hand/arm inpainting Stage-1 QA

Status: **FAIL — TELEA reconstruction is not sufficiently naturalistic for full-scale use.**

## Population

- 48 fixed ST context-only images.
- 0 primary-test images.
- No classifier predictions used.

## Executed pipeline

The recovered historical hand/arm cleaning composition was restored with current-runtime compatibility:

- SCHP Pascal-7 arm segmentation through the pinned ONNX export;
- MediaPipe hand detection through the repository-verified HandLandmarker Tasks model;
- unioned arm + hand masks;
- historical closing/dilation;
- half-resolution OpenCV TELEA radius 5;
- cubic resize;
- feathered compositing.

Run: `36983299845`.

Artifact: `11216458307`.

Artifact SHA256: `90a898a946160c8bc128140228dc466bccca68c283d459958263818cde8ccdbd`.

## Numerical mask summary

- images: 48;
- zero combined masks: 0;
- zero arm masks: 11;
- zero hand masks: 2;
- mean final mask fraction: 0.13345;
- median: 0.11190;
- p90: 0.23207;
- maximum: 0.24138.

## Visual QA result

**Fail.**

The reconstructed regions frequently become large smooth/polygonal patches and do not resemble plausible underlying page/background content. The failure is visible across all four before/after contact sheets, particularly when the removed region occupies a large portion of the page.

The historical full-corpus LRASPP + TELEA QA artifact from run `36145174745` was also inspected. It shows the same qualitative failure mode: large artificial fill regions after person-mask inpainting.

Therefore the failure is not treated as evidence that the SCHP/MediaPipe mask itself is invalid. The concrete defect is that **local TELEA inpainting is unsuitable for these large structured occlusions**.

## Consequence

- Do **not** promote TELEA output to the frozen 100-image PoC.
- Do **not** build a 2,989-image TELEA-cleaned corpus.
- Do **not** evaluate any classifier on these failed reconstructions.

## Prospectively frozen correction

Keep the hand/arm mask definition fixed and change only the reconstruction engine.

Next candidate: **video-assisted same-video reconstruction**.

For each target frame:

1. use nearby frames from the same canonical video;
2. compute the same hand/arm masks on candidate source frames;
3. align candidate source to target using feature-based homography on unmasked content;
4. accept only geometrically/photometrically credible alignments;
5. fill target hand/arm pixels only from aligned source pixels that are themselves unmasked;
6. combine multiple accepted sources when available;
7. use TELEA only for any small residual hole after real-pixel transfer;
8. feather the final seam.

This candidate is selected solely because Stage-1 visual QA exposed a reconstruction defect. No classifier result is used for method choice.
