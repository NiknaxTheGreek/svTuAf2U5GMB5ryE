# Phase 11B — Chroma QA Stage 1

Status: **PARTIAL PASS — naturalistic transform is visually acceptable but quantitatively too weak.**

Run: `37073160426`.

Artifact: `11255842392`.

Artifact SHA256:
`f95810847ea12537f349ce4cc44246ba235dceb314f5d95aab46925cb8d4a162`.

## Population

- 100 frozen ST-context development images.
- 0 primary-test images.
- 0 classifier predictions.
- Donor selection was label-independent.
- Universal donor pool contained 1,428 images that were non-test in O, S, T and ST.
- Same-video donors: 0.
- Same-environment donors: 0.

The post-selection same-label fraction was 0.51, consistent with label not being used for donor choice.

## Matched +90° chroma rotation

- mean Rec.709 luminance MAE: 0.1706;
- 95th percentile per-image luminance MAE: 0.1971;
- mean gamut-compressed pixel fraction: 0.0311;
- frozen luminance gate: **PASS**.

## Naturalistic palette transfer, strength 0.75

- mean Rec.709 luminance MAE: 0.1849;
- 95th percentile per-image luminance MAE: 0.2553;
- mean gamut-compressed pixel fraction: 0.0136;
- median per-image chroma displacement: **1.5537**.

Luminance gate: **PASS**.

Non-triviality gate (median chroma displacement >= 3.0): **FAIL**.

## Visual review

All 100 outputs were reviewed through ten contact sheets.

The naturalistic counterfactual preserved page structure, text layout, edges and overall scene plausibility. No recurring major colour artifact pattern was observed. Unlike the failed hand/arm inpainting experiments, the transform does not create synthetic geometry or missing-content artifacts.

The failure is therefore **insufficient perturbation strength**, not visual implausibility.

## Frozen correction

Before any primary test image is transformed, rerun the identical deterministic procedure with only one change:

- palette-transfer strength: **0.75 -> 1.00**.

Everything else remains frozen:

- same 100 development images;
- same universal donor pool;
- same deterministic donor assignment;
- same covariance transform;
- same luminance-preserving gamut compression;
- same quantitative thresholds.

If strength 1.00 still fails the non-triviality gate, do not tune further by observing primary-test performance.
