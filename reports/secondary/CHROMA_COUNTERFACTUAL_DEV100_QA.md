# Phase 11B — Chroma counterfactual development QA (first 100)

Status: **matched control PASS; naturalistic v1 FAILS non-triviality gate**.

Run: `37073160426`.

Artifact: `11255842392`.

Artifact SHA256:
`f95810847ea12537f349ce4cc44246ba235dceb314f5d95aab46925cb8d4a162`.

## Population

- 100 frozen ST context-only images.
- 0 primary-test images.
- 0 classifier predictions.
- Universal donor pool: 1,428 images that are non-test simultaneously in O, S, T and ST.
- Donor selection did not use the target label.
- Same-video donors: 0.
- Same-environment donors: 0.
- Same-label fraction after label-independent selection: 0.51.

## Matched +90 degree chroma rotation

- mean Rec.709 luminance MAE: 0.1706;
- p95 per-image luminance MAE: 0.1971;
- mean gamut-compressed pixel fraction: 0.0311.

The pre-frozen luminance gate passes.

Visual review shows a strong colour intervention while preserving page geometry, text layout and luminance structure. This variant is accepted as the matched chroma-distribution control.

## Naturalistic palette transfer v1

- strength: 0.75;
- mean Rec.709 luminance MAE: 0.1849;
- p95 per-image luminance MAE: 0.2553;
- mean gamut-compressed pixel fraction: 0.0136;
- median per-image chroma displacement: **1.5537**.

Luminance preservation passes, but the pre-frozen non-triviality requirement was median chroma displacement >= 3.0.

Therefore naturalistic v1 fails the quantitative gate because it is **too weak**, not because it damages the image.

Visual review is consistent with this: outputs are generally plausible but frequently almost indistinguishable from the original.

## Development decision

No primary test is opened.

One context-only correction is allowed:

- increase palette-transfer strength to 1.0;
- choose the donor from a deterministic label-independent shortlist of real universal-pool palettes by maximum chroma-statistics distance;
- validate on a new, non-overlapping 100-image ST-context set.

The first 100 images are now development-only and cannot serve as the validation set for naturalistic v2.

If naturalistic v2 fails the independent validation gate, naturalistic colour counterfactual development stops.
