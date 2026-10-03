# Phase 11B — Naturalistic chroma strength amendment

The initial frozen naturalistic transfer at strength 0.75 completed on 100 ST-context images and preserved luminance, but failed only the pre-specified non-triviality gate:

- mean Rec.709 luminance MAE: 0.1849;
- p95 per-image luminance MAE: 0.2553;
- median per-image chroma displacement: 1.5537;
- required median chroma displacement: >=3.0.

Visual review found the outputs generally plausible but too similar to the originals.

Accordingly, before any primary-test evaluation, a single controlled context-only strength escalation is frozen.

Candidate strengths:

- 1.0;
- 1.5.

Selection rule:

1. both candidates use the identical universal label-independent donor pool and deterministic donor assignment already frozen;
2. both must satisfy the existing luminance gates;
3. median per-image chroma displacement must be >=3.0;
4. visual major-artifact fraction must be <=10% across the same 100 context images;
5. choose the **lowest** candidate strength satisfying all gates;
6. if neither passes, reject the naturalistic colour counterfactual rather than tuning further.

No classifier prediction and no O/S/T/ST primary-test image may be used in this amendment.
