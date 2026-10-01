# Phase 11B — Hand/arm inpainting counterfactual

Status: **prospectively frozen secondary/exploratory PoC**.

The earlier hand-region experiment remains valid as an **occlusion sensitivity** analysis. It is not treated as realistic hand removal.

## Scientific question

What happens to frozen Best-O C14 when visible hand/arm content is removed and the affected document/background region is reconstructed to look plausibly as though the limb had never been present?

## PoC population

Use exactly 100 deterministic images from **ST context-only** rows. No primary O/S/T/ST test image may enter method development.

Selection is balanced across the non-empty `(role_detail, environment_id, label)` strata and stable-hash ordered within each stratum.

## Mask requirement

The target mask must cover the visible human limb relevant to the counterfactual:

- fingers;
- hand/palm;
- wrist;
- visible forearm/arm where present.

MediaPipe hand landmarks are only the initial seed. The PoC may use deterministic skin-connected extension and wrist-to-boundary extension. Visual QA is mandatory.

Manual correction may be used only to diagnose/develop the PoC. A full-scale classifier experiment requires a frozen automatic mask recipe first.

## Candidate inpainting methods

Compare a deliberately small fixed set:

1. OpenCV TELEA, radius 5 px;
2. OpenCV Navier–Stokes, radius 5 px;
3. pretrained LaMa, if it runs reproducibly in the controlled environment.

Classifier predictions must not participate in choosing the inpainting method.

## PoC quality gate

Review all 100 development outputs before any classifier evaluation.

A method must satisfy, descriptively and by recorded review:

- at least 90% of PoC images have the visible hand/arm removed adequately;
- no more than 10% show a major visible inpainting artifact;
- document geometry remains plausible;
- text/page/background outside the target region is preserved;
- changes outside the mask are negligible except a narrow seam/feather ring.

If no method passes, full-scale inpainting stops and the failed PoC is reported as a valid negative result.

## Full-scale evaluation if the gate passes

Freeze the automatic mask + inpainting pipeline, then evaluate frozen Best-O C14 on identical O-test images:

- original;
- inpainted no-hand/arm counterfactual.

Retain the already-completed hand-mask and matched-occlusion results as controls.

Use threshold 0.5 and 5,000 paired frame plus 5,000 paired video/group bootstrap resamples. This analysis is post-hoc and cannot change Best O.
