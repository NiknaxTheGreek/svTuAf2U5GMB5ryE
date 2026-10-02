# Phase 11B — Video-assisted reconstruction Stage-2 QA

Status: **FAIL — insufficient real-pixel coverage; residual TELEA remains visually unacceptable.**

## Population

- Same fixed 48 ST context-only targets.
- Source frames restricted to ST context-only rows from the same canonical video.
- 0 primary-test images.
- 0 classifier predictions.

Run: `36984523373`.

Artifact: `11217245472`.

Artifact SHA256: `fc70541323e5b7692177970d3c306a14a82a9413175849062379c1f529bb2b8a`.

## Frozen quantitative gate result

- 46/48 targets had at least one accepted same-video alignment.
- Mean accepted sources: 5.27.
- Median accepted sources: 5.
- Only 5/48 images met the pre-frozen >=80% real-pixel / <=20% residual gate.
- Fraction passing: 0.1042.
- Mean real-pixel fill: 0.2506.
- Median real-pixel fill: 0.1748.
- Mean residual fraction: 0.7077.
- Median residual fraction: 0.8076.

Therefore the quantitative gate failed.

## Visual QA

Real aligned neighboring-frame pixels are often plausible where available, confirming that same-video reconstruction can recover useful content.

However, most masked pixels remain uncovered by unobstructed aligned neighbors. The large residual regions are then handled by TELEA and retain the same visible smooth/polygonal artifacts identified in Stage 1. Several reconstructed images therefore remain unsuitable as naturalistic counterfactuals.

## Consequence

- Do not promote video+TELEA to the frozen 100-image gate.
- Do not build regime train/test datasets from these outputs.
- Preserve video-derived real pixels as a potentially useful first-stage fill.

## Next frozen correction

Evaluate a large-mask generative inpainting engine on the same 48 context images:

1. **LaMa full** — LaMa fills the entire frozen hand/arm mask.
2. **Video + LaMa residual** — accepted aligned same-video pixels fill whatever they can; LaMa fills only the remaining mask.

The two variants use identical frozen masks. Selection is by context-only visual/preservation QA, never classifier performance.
