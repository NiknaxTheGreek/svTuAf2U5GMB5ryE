# Phase 11B — Chroma counterfactual QA v1

Status: **PARTIAL PASS / NATURALISTIC COUNTERFACTUAL UNDERPOWERED**.

Run: `37073160426`.

Artifact: `11255842392`.

Artifact SHA256:
`f95810847ea12537f349ce4cc44246ba235dceb314f5d95aab46925cb8d4a162`.

## Population

- frozen 100-image ST-context sample;
- 0 primary-test images;
- 0 classifier predictions;
- universal donor pool: 1,428 images that are non-test in O, S, T and ST simultaneously;
- target labels were not used for donor selection;
- same-video donors: 0;
- same-environment donors: 0.

The post-selection same-label fraction was 0.51, consistent with label-independent donor choice rather than enforced same-class matching.

## Matched +90 degree chroma rotation

Quantitative luminance gate: **PASS**.

- mean Rec.709 luminance MAE: 0.1706;
- p95 per-image luminance MAE: 0.1971;
- mean gamut-compressed fraction: 0.0311.

The contact sheets confirm the intended strong hue perturbation while geometry/text layout remain fixed.

## Naturalistic palette transfer v1

Luminance preservation: **PASS**.

- mean Rec.709 luminance MAE: 0.1849;
- p95 per-image luminance MAE: 0.2553;
- mean gamut-compressed fraction: 0.0136.

Visual plausibility: **PASS qualitatively** on review of the 100-image contact sheets. The transformed images remain document-like, with preserved geometry, text structure and plausible skin/page colours.

Counterfactual strength gate: **FAIL**.

- median per-image chroma displacement: 1.5537;
- required minimum: 3.0.

The naturalistic transform is therefore too subtle to serve as a strong colour counterfactual even though it is visually plausible.

## Frozen correction before any test evaluation

No O/S/T/ST test image has been transformed or evaluated.

A second context-only QA is allowed because the failure is purely insufficient perturbation strength.

The v2 naturalistic counterfactual will:

1. preserve the same universal non-test donor pool;
2. never use target labels or classifier predictions;
3. construct a deterministic candidate donor shortlist by SHA256;
4. within that fixed shortlist, choose the donor with the largest target-vs-donor chroma Gaussian distance;
5. require a different canonical video and environment where possible;
6. use full-strength (1.0) 2-D chroma covariance transfer;
7. retain target luminance exactly via the same radial gamut compression.

The revised transform must again pass the full 100-image luminance, non-triviality and visual plausibility gates before any primary test evaluation.
