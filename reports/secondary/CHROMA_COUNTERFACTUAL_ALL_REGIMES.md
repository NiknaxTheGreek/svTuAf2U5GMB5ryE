# Phase 11B — Frozen-champion chroma counterfactual results

Post-hoc, non-selection-bearing sensitivity analysis. The frozen O/S/T/ST champions remain unchanged.

| Regime | Variant | F1 | ΔF1 vs original | Video/group 95% CI | Changed predictions |
|---|---|---:|---:|---:|---:|
| O | original | 0.9983 | +0.0000 | — | 0 |
| O | grayscale | 0.6840 | -0.3143 | [-0.4130, -0.2303] | 269 |
| O | matched_rotation | 0.6539 | -0.3444 | [-0.4477, -0.2559] | 308 |
| O | naturalistic_v2 | 0.9386 | -0.0596 | [-0.0940, -0.0344] | 36 |
| S | original | 0.8196 | +0.0000 | — | 0 |
| S | grayscale | 0.6225 | -0.1971 | [-0.3488, -0.0529] | 425 |
| S | matched_rotation | 0.6225 | -0.1971 | [-0.3488, -0.0529] | 425 |
| S | naturalistic_v2 | 0.7310 | -0.0887 | [-0.2018, +0.0012] | 179 |
| T | original | 0.8800 | +0.0000 | — | 0 |
| T | grayscale | 0.6695 | -0.2105 | [-0.3029, -0.1266] | 363 |
| T | matched_rotation | 0.6695 | -0.2105 | [-0.3029, -0.1266] | 363 |
| T | naturalistic_v2 | 0.8224 | -0.0576 | [-0.1193, -0.0042] | 102 |
| ST | original | 0.7040 | +0.0000 | — | 0 |
| ST | grayscale | 0.7109 | +0.0069 | [-0.2233, +0.2444] | 86 |
| ST | matched_rotation | 0.7246 | +0.0206 | [-0.1089, +0.1738] | 23 |
| ST | naturalistic_v2 | 0.5818 | -0.1222 | [-0.2919, +0.0307] | 50 |

## Interpretation framework

- Grayscale is the crude chroma-removal ablation.
- Matched +90° chroma rotation is a strong colour-distribution control that preserves luminance/spatial structure.
- Naturalistic v2 is the independently validated plausible RGB counterfactual with target-label-independent donor selection.
- A naturalistic-v2 effect provides stronger evidence that chromatic cues influence the frozen model than grayscale alone.

No result in this report can redefine Best O/S/T/ST.
