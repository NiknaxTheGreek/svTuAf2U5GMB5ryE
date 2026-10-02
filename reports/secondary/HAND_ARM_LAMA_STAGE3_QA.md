# Phase 11B — LaMa Stage-3 QA

Status: **FAIL — improved over TELEA, but still not sufficiently naturalistic for promotion.**

## Population and controls

- 48 fixed ST context-only development images.
- 0 primary-test images.
- 0 classifier predictions.
- Frozen hand/arm masks identical across variants.
- Outside-mask pixels were forced to remain exactly original.

Run: `36985589884`.

Artifact: `11217248702`.

Artifact SHA256: `246ca4740c6b39890f17fb8232f7e2fc99239bd78918dd4fc906a2d9bdee154b`.

Pinned LaMa model SHA256:
`7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c`.

## Variants

1. **LaMa full** — LaMa fills the entire frozen hand/arm mask.
2. **Video + LaMa residual** — accepted real same-video pixels fill first; LaMa fills the remaining mask.

## Numerical preservation checks

- LaMa full: 0/48 images changed outside the target mask.
- Video + LaMa: 0/48 images changed outside the target mask.
- Mean real-pixel contribution to the hybrid: 0.2975.
- Median real-pixel contribution: 0.1863.

## Visual QA

Both variants are materially better than large-mask TELEA, but neither passes the pre-frozen naturalistic gate.

Common failure modes include:

- blurred or flesh-toned smears where a hand/arm was removed;
- implausible reconstruction of text-heavy page regions;
- ghost-like hand/forearm structure in some large masks;
- hybrid outputs that inherit imperfect real-pixel transfer and still require large generative residuals.

The major-artifact rate is visibly above the allowed 10% threshold.

## Consequence

- Do not promote either LaMa variant to the frozen 100-image gate.
- Do not build O/S/T/ST cleaned train/test sets from these outputs.
- Preserve LaMa only as a residual-fill component if a later temporal mosaic reduces the missing region enough.

## Next frozen correction

One final video-derived reconstruction PoC is justified because the previous Stage-2 implementation capped source candidates at eight nearest context frames.

The next PoC uses:

- **all same-video ST non-test frames** (train + context; primary ST test forbidden);
- the same frozen alignment acceptance thresholds;
- sharp **best-source-per-pixel** transfer rather than weighted averaging;
- source ranking by alignment quality with temporal-distance penalty;
- LaMa only for residual pixels not recovered from real aligned source frames.

If this still leaves large residuals or fails visual QA, the realistic hand/arm-cleaning extension will be reported as a failed PoC rather than forced into the full experiment.
