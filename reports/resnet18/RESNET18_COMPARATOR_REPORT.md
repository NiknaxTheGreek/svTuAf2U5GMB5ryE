# Fixed ResNet18 vs frozen scratch champions

Paired bootstrap comparison on identical saved test samples. Positive differences favor ResNet18. All analysis is post-hoc and non-selection-bearing.

| Regime | Scratch | Scratch F1 | ResNet18 F1 | ΔF1 | Frame 95% CI | Video/group 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| O | C14 | 0.9983 | 0.9931 | -0.0052 | -0.013–+0.002 | -0.014–+0.002 |
| S | C18 | 0.8196 | 0.7807 | -0.0390 | -0.074–-0.004 | -0.169–+0.076 |
| T | C06 | 0.8800 | 0.8807 | +0.0007 | -0.021–+0.023 | -0.032–+0.030 |
| ST | C07 | 0.7040 | 0.7196 | +0.0156 | -0.093–+0.127 | -0.212–+0.259 |

The fixed ResNet18 recipe is a comparator, not a replacement selection procedure. Interpret a difference cautiously when the video/group interval includes zero.
