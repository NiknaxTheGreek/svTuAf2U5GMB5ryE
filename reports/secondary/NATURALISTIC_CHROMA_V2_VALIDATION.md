# Phase 11B — Naturalistic chroma v2 independent validation

Status: **PASS**.

Run: `37073967071`.

Artifact: `11255238172`.

Artifact SHA256:
`953ae8bc50833f72c6bbb452cdee9ce2901d1318f2318a7ea3d4fcf4e2220af5`.

## Validation population

- 100 deterministic ST-context images.
- 0 overlap with the first 100-image development set.
- 0 primary-test overlap.
- 0 classifier predictions used.
- Frozen validation sample SHA256:
  `fa9a78b8448d5a8620a0ef501d7834a3d99f199201b7d4d4d81d2814efaaf00a`.

## Donor independence

- universal donor pool: 1,428 images non-test simultaneously in O/S/T/ST;
- donor shortlist: 16 deterministic label-blind candidates per target;
- chosen donor: maximum real-palette chroma-statistics distance within shortlist;
- same-video donors: 0;
- same-environment donors: 0;
- same-label fraction after selection: 0.50;
- target label used for donor selection: false.

## Quantitative validation

Naturalistic v2:

- strength: 1.0;
- mean Rec.709 luminance MAE: 0.1876;
- p95 per-image luminance MAE: 0.2397;
- median per-image chroma displacement: 3.7516;
- mean gamut-compressed pixel fraction: 0.0261.

Pre-frozen requirements:

- mean luminance MAE <= 0.5: PASS;
- p95 per-image luminance MAE <= 1.0: PASS;
- median chroma displacement >= 3.0: PASS.

## Visual validation

All 100 validation contact-sheet examples were reviewed.

The transformation changes palette/chroma while preserving:

- page geometry;
- text layout;
- spatial detail;
- luminance structure.

Unlike the failed hand/arm reconstructions, this operation does not synthesize geometry or missing content. The observed recolourings remain plausible enough for the intended counterfactual role, and the major visual-artifact fraction is below the pre-frozen 10% limit.

Visual gate: **PASS**.

## Frozen chroma variants for primary sensitivity evaluation

The O/S/T/ST frozen champions may now be evaluated on:

1. original RGB;
2. Rec.709 grayscale;
3. matched +90-degree chroma rotation;
4. naturalistic palette counterfactual v2.

No retraining, model selection or threshold changes are permitted.
