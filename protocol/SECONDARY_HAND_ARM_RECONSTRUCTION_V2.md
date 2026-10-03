# Phase 11B — Hand/Arm Reconstruction v2

Status: **FROZEN DEVELOPMENT METHOD — context-only PoC continuation**.

This document opens a new reconstruction family after the first-generation TELEA / same-video mosaic / LaMa family failed its prospectively frozen PoC gate. The earlier negative result remains valid and is not rewritten.

## Scientific role

- Secondary/exploratory only.
- Frozen O/S/T/ST champions remain unchanged.
- No classifier prediction may be used to choose a donor, alignment, threshold, or reconstruction.
- No primary-test image may be inspected or used while developing or tuning this method.
- The 48-image ST-context development sample remains development-only.
- Any later validation must use a new non-overlapping context-only sample.

## Why v2 is materially different

The first-generation method aligned complete frames and restricted reconstruction sources to the same video. That left large regions unrecovered when the hand/arm occupied the same page location for most or all frames.

v2 treats the photographed object as a document plane and uses two additional sources of structure:

1. **document-plane normalization** before detailed alignment;
2. **label-blind cross-video donor retrieval** from ST non-test images, followed by strict geometric and photometric verification.

The goal is to recover obscured page pixels from actual observed document pixels whenever the same page/content appears elsewhere in the eligible non-test corpus.

## Frozen v2 pipeline

1. Detect hand/arm using the pinned SCHP arm segmenter plus verified MediaPipe Tasks hand detector.
2. Detect a document/book quadrilateral when possible and normalize it to a canonical document view.
3. Build a deterministic appearance descriptor from normalized grayscale + gradient structure.
4. Candidate donors:
   - all eligible same-video ST non-test frames;
   - top-K different-video ST non-test candidates ranked only by the normalized appearance descriptor.
5. For each candidate:
   - estimate document-plane source→target homography;
   - refine with ORB correspondences outside target and donor hand/arm masks;
   - reject weak geometric support;
   - reject low clean overlap;
   - reject high outside-mask grayscale disagreement.
6. Fuse accepted real donor pixels inside the target hand/arm region using a per-pixel observed-color medoid around the donor median. This preserves an actually observed RGB donor value rather than generating a new median color.
7. Record per-pixel support as a confidence map.
8. Run LaMa **only** on residual masked pixels for which no accepted real donor observation exists.
9. Restore every outside-mask target pixel exactly.

Same-video donors may use the historical alignment routine only as a fallback when document quadrilateral estimation is unavailable. Cross-video donors require document-plane verification.

## Frozen development gate

For each target:

- real observed-donor coverage >= 0.80;
- residual generative coverage <= 0.20.

Across the 48 development targets:

- >= 90% of targets must meet both coverage criteria;
- zero outside-mask pixel changes;
- visual QA must pass.

The visual gate specifically rejects visible hand remnants, flesh-toned smears, obvious blur/seams, damaged document geometry, and invented/corrupted text where donor evidence should have recovered the page.

## Promotion rule

If and only if the quantitative and visual development gates pass:

1. freeze the exact v2 config and code;
2. generate a new deterministic 100-image ST-context validation population with zero overlap with the 48-image development set and zero primary-test overlap;
3. run the identical frozen reconstruction;
4. require the same quantitative and visual standards.

Only after independent validation passes may a larger naturalistic hand/arm-cleaned corpus or cleaned-domain secondary retraining experiment be considered.

## Historical result

The previous naturalistic reconstruction family remains recorded as a failed first-generation PoC. v2 is a new method family; it does not retrospectively turn that earlier run into a pass.
