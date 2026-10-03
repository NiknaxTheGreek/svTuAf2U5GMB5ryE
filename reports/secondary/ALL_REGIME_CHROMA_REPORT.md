# Phase 11C — All-regime chroma sensitivity

Post-hoc frozen-model analysis. No retraining, threshold changes, or champion replacement.

| Regime | Original F1 | Grayscale F1 (Δ) | Matched +90 F1 (Δ) | Naturalistic v2 F1 (Δ) |
|---|---:|---:|---:|---:|
| O | 0.9983 | 0.6840 (-0.3143) | 0.6539 (-0.3444) | 0.9386 (-0.0596) |
| S | 0.8196 | 0.6225 (-0.1971) | 0.6225 (-0.1971) | 0.7310 (-0.0887) |
| T | 0.8800 | 0.6695 (-0.2105) | 0.6695 (-0.2105) | 0.8224 (-0.0576) |
| ST | 0.7040 | 0.7109 (+0.0069) | 0.7246 (+0.0206) | 0.5818 (-0.1222) |

## Video/group paired-bootstrap intervals

### O
- rec709_grayscale: [-0.4130, -0.2303]
- matched_chroma_rotation: [-0.4477, -0.2559]
- naturalistic_v2: [-0.0940, -0.0344]

### S
- rec709_grayscale: [-0.3512, -0.0561]
- matched_chroma_rotation: [-0.3512, -0.0561]
- naturalistic_v2: [-0.2030, -0.0007]

### T
- rec709_grayscale: [-0.3047, -0.1244]
- matched_chroma_rotation: [-0.3047, -0.1244]
- naturalistic_v2: [-0.1202, -0.0058]

### ST
- rec709_grayscale: [-0.2157, +0.2401]
- matched_chroma_rotation: [-0.1073, +0.1753]
- naturalistic_v2: [-0.2911, +0.0270]

