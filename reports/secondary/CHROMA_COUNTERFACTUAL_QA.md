# Phase 11C — Chroma counterfactual QA

Status: **PASS**.

Run: `37019025030`.

Evidence artifact: `11232542221`.

Artifact SHA256: `e836287cbd29e70424d7a5701f6f1e67e63d2f273bab17785e93e3b37e76aadb`.

## Development population

- 100 frozen deterministic ST context-only images.
- 0 primary-test images.
- 0 classifier predictions.
- Donor palette selection never used the target label.
- All 100 naturalistic donors came from a different environment.

## Quantitative QA

| Transform | Mean luminance MAE | p95 luminance error | RGB gamut clipping | Gate |
|---|---:|---:|---:|---|
| Matched +90° chroma rotation | 0.1704 | 0.3582 | 1.035% | PASS |
| Naturalistic palette counterfactual | 0.1799 | 0.3622 | 0.578% | PASS |

Both transforms preserve Rec.709 luminance far inside the frozen tolerances.

No spatial resampling or geometry modification occurs.

## Visual QA

All ten QA contact sheets covering all 100 development images were reviewed.

### Matched chroma rotation

The transform produces the intended conspicuous hue redistribution while keeping document layout, text, edges, illumination structure and geometry intact. It is intentionally not required to look natural; its role is the matched colour-distribution perturbation control.

### Naturalistic palette counterfactual

**PASS.**

Across the 100 reviewed context images, the recoloured images remain plausible RGB photographs of the same book/page/environment. The transform changes colour subtly while retaining:

- page geometry;
- text and figure structure;
- hand/arm geometry;
- shadows and luminance structure;
- scene layout.

No obvious false-colour or spatial artifacts were observed at a rate approaching the frozen 10% failure threshold.

## Promotion

Both frozen transforms may now be applied to the sealed O/S/T/ST test memberships for **frozen-model sensitivity evaluation**.

The four original champions remain unchanged:

- O C14;
- S C18;
- T C06;
- ST C07.

The sensitivity stage uses no retraining and cannot replace a primary champion.
