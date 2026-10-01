# Phase 11 — Best-O perturbation sensitivity

Post-hoc frozen-model analysis only; no retraining or champion change.

| Variant | F1 | Precision | Recall | Accuracy | ΔF1 vs original |
|---|---:|---:|---:|---:|---:|
| original | 0.9983 | 1.0000 | 0.9966 | 0.9983 | +0.0000 |
| grayscale | 0.6848 | 0.5206 | 1.0000 | 0.5528 | -0.3135 |
| hand_mask | 0.9228 | 0.8645 | 0.9897 | 0.9196 | -0.0754 |
| matched_control | 0.9051 | 0.8363 | 0.9862 | 0.8995 | -0.0932 |

Hand-mask minus matched-control ΔF1 contrast, video/group 95% CI: [-0.022090536418955167, 0.0651733747884311].

Evidence for hand-specific shortcut dependence requires hand-mask degradation materially beyond matched equal-area control, preferably with video-group bootstrap support. This detector is not validated complete human segmentation.
