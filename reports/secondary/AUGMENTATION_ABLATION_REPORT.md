# Phase 11 — Fixed mild augmentation ablation

This is a **post-hoc, non-selection-bearing** ablation. The original frozen champions remain Best O/S/T/ST.

| Regime | Frozen config | Original F1 | Augmented retrain F1 | ΔF1 | Paired video/group 95% CI |
|---|---:|---:|---:|---:|---:|
| O | C14 | 0.9983 | 0.9617 | -0.0365 | -0.0632 to -0.0168 |
| S | C18 | 0.8196 | 0.5812 | -0.2385 | -0.3839 to -0.0983 |
| T | C06 | 0.8800 | 0.7377 | -0.1423 | -0.1978 to -0.0938 |
| ST | C07 | 0.7040 | 0.7576 | +0.0536 | +0.0071 to +0.1228 |

All four video/group intervals exclude zero.

The fixed mild augmentation recipe therefore had **heterogeneous effects**:

- O degraded modestly;
- S degraded strongly;
- T degraded materially;
- ST improved.

This does not justify replacing any frozen primary champion. It shows that the same geometric/photometric augmentation is not uniformly beneficial across the four generalization regimes.

The result is selection-conditioned and post-hoc because the regime champions were already known before this ablation.
