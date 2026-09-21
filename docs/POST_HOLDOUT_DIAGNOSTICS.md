# Post-Holdout Error and Robustness Analysis

## Status

Step 8 post-holdout diagnostics: COMPLETE.

This step is strictly analysis-only. The final reported Step 7 model, epoch, threshold, and one-shot test metrics are unchanged.

Governance lock:

- model changed: no
- threshold changed: no
- post-test tuning performed: no
- final test metrics changed: no
- holdout status: consumed

## Final test result retained

The immutable one-shot final test result remains:

- F1: 0.7514450867
- Precision: 0.8515283843
- Recall: 0.6724137931
- Accuracy: 0.7839195980
- TN: 273
- FP: 34
- FN: 95
- TP: 195

## Source-group uncertainty

Because frames within one clip are correlated, frame-level sample size overstates independent information. A deterministic 20,000-replicate stratified source-group bootstrap was therefore run across the 13 flip and 11 notflip test groups.

95% bootstrap intervals:

| Metric | Lower | Median | Upper |
|---|---:|---:|---:|
| F1 | 0.6142 | 0.7536 | 0.8829 |
| Precision | 0.6920 | 0.8599 | 0.9888 |
| Recall | 0.5048 | 0.6764 | 0.8558 |
| Accuracy | 0.6601 | 0.7867 | 0.8996 |

The wide intervals are consistent with substantial between-clip variability and the relatively small number of independent test clips.

## Error concentration and sequence behavior

Of the 24 test source groups:

- 11 have zero frame errors.
- 13 contain at least one error.
- Median binary prediction transitions within a group: 1.
- Maximum transitions within a group: 8.
- Longest consecutive error run: 15 frames.

Largest failure groups:

| Source group | Label | Frames | Errors | Error rate |
|---|---|---:|---:|---:|
| flip__0038 | flip | 46 | 32 | 69.6% |
| notflip__0053 | notflip | 30 | 23 | 76.7% |
| flip__0041 | flip | 20 | 18 | 90.0% |
| flip__0042 | flip | 30 | 16 | 53.3% |
| flip__0047 | flip | 25 | 11 | 44.0% |
| flip__0045 | flip | 24 | 8 | 33.3% |

These failures are persistent over portions of clips rather than isolated random mistakes.

## Confidence diagnostic

Prediction margin is defined as `abs(prob_flip - 0.5)`.

Mean margin:

- Correct frames: 0.3898
- Incorrect frames: 0.2612

Near-threshold predictions are much less reliable:

- Margin 0-0.05: 20 frames, 60.0% error rate
- Margin 0.05-0.10: 31 frames, 35.5% error rate
- Margin 0.10-0.20: 58 frames, 46.6% error rate
- Margin 0.20-0.30: 73 frames, 32.9% error rate
- Margin 0.30-0.40: 86 frames, 30.2% error rate
- Margin 0.40-0.50: 329 frames, 8.8% error rate

Higher confidence is associated with fewer errors overall, but 29 high-margin frames are still wrong, so confidence is not a sufficient error detector.

## Frame-position diagnostic

Error rate by normalized position inside each source clip:

| Position quintile | All frames | Flip frames | Notflip frames |
|---|---:|---:|---:|
| Q1 | 23.8% | 37.5% | 9.7% |
| Q2 | 12.8% | 17.9% | 8.2% |
| Q3 | 20.9% | 33.3% | 9.8% |
| Q4 | 23.7% | 40.4% | 8.2% |
| Q5 | 26.4% | 33.9% | 19.4% |

Flip errors are elevated throughout much of the clip, especially Q1 and Q4. Notflip errors are lower overall but rise at the end of clips. These are descriptive post-holdout findings only and are not used to alter labels, threshold, or model.

## Fixed perturbation robustness battery

The exact frozen epoch-4 model and threshold 0.5 were evaluated under six fixed image perturbations after the holdout had already been consumed.

| Condition | F1 | Delta vs original |
|---|---:|---:|
| Original final test | 0.75145 | — |
| Brightness ×0.85 | 0.75836 | +0.00692 |
| Contrast ×0.85 | 0.75049 | -0.00095 |
| Gaussian blur radius 1 | 0.74903 | -0.00241 |
| JPEG quality 70 | 0.74319 | -0.00825 |
| Contrast ×1.15 | 0.70281 | -0.04863 |
| Brightness ×1.15 | 0.69636 | -0.05509 |

The largest observed degradation is under increased brightness and increased contrast, mainly through lower flip recall. Mild blur and JPEG recompression have comparatively small effects.

The small F1 increase under brightness ×0.85 is not a tuning result and must not be used to replace the reported preprocessing or final test score.

## Interpretation

The principal limitation is source-group generalization rather than generic corruption sensitivity.

Evidence:

1. Test performance varies strongly by source clip.
2. Error runs persist for several consecutive frames within difficult clips.
3. Source-group bootstrap uncertainty is wide.
4. Mild blur and JPEG compression have little impact compared with the much larger clip-to-clip variation.
5. Increased brightness/contrast can further reduce recall, indicating some sensitivity to photometric conditions.

This motivates the downstream sequence-level extension and future data-collection/domain-robustness work, but it does not justify modifying the reported single-image model after test access.

## Reproducibility

Successful diagnostic workflow run: `35653218914`.

Artifact hashes:

- `source_group_diagnostics.csv`: `1b368517605c609fe5c72b46edbab43eea9388c6fe20a011a502d1a7983c3741`
- `frame_position_diagnostics.csv`: `54fb03cc160a9aa1f14d6836fc019088aedfae0db9768073625299bfb9c768cc`
- `perturbation_robustness.csv`: `0d1f76886590b279966ef9762e59cdebdb5507d64a7f203a4346648df22bb06b`

The workflow verifies the immutable Step 6 checkpoint and immutable Step 7 final predictions by SHA-256 before any analysis.
