# MonReader V2 — Source-Safe (S) Scratch-CNN Selection Benchmark

## Question

How well does the frozen C01–C20 scratch-CNN candidate family perform when the **entire acquisition environment ENV-03 is absent from training**?

S is the first primary source-disjoint benchmark in the V2 rebuild. The 770-image S test contains 29 videos from ENV-03, while training contains 2,219 images from 88 videos in ENV-01, ENV-02 and ENV-04. Canonical train/test video overlap is zero.

Because the S test is used to select among 20 already-frozen candidates, the winning result is a **source-disjoint selection-benchmark estimate**, not an untouched post-selection estimate.

## Execution integrity

For S:

- training rows: 2,219;
- test rows: 770;
- training environments: ENV-01, ENV-02, ENV-04;
- held-out environment: ENV-03;
- training videos: 88;
- test videos: 29;
- train/test video overlap: 0;
- threshold: 0.5;
- exactly 20 epochs per candidate;
- validation rows loaded during candidate training: 0;
- test rows loaded during candidate training: 0;
- all 20 candidate jobs completed successfully;
- all 20 final epoch-20 checkpoints were hash-verified before S test construction;
- every candidate received exactly one S-test evaluation.

The closed-test checkpoint registry reported `PASS_ALL_20_FROZEN_BEFORE_S_TEST` with `test_evaluation_opened=false`.

An earlier Phase-5 workflow attempt invoked the O-specific trainer from the S workflow. Those candidate jobs failed at argument parsing before S model training and before S test evaluation. The authoritative corrected run changed only the module invocations. No scientific result from the failed run is used.

## Best S

**C18** is Best S by the predeclared ordering.

Configuration:

- depth: 4 convolution blocks;
- start filters: 12;
- dropout: 0.193994;
- optimizer: Adam;
- learning rate: 0.000170852;
- weight decay: 6.47898e-06;
- batch size: 78;
- trainable parameters: 55,213;
- final epoch-20 training loss: 0.161004.

### Held-out ENV-03 metrics

| Metric | Best S (C18) |
|---|---:|
| F1 | **0.819625** |
| Precision | 0.823188 |
| Recall | 0.816092 |
| Accuracy | 0.837662 |
| Balanced accuracy | 0.835771 |
| ROC-AUC | 0.900236 |
| PR-AUC | 0.899480 |
| TN / FP / FN / TP | 361 / 61 / 64 / 284 |

This is materially lower than the near-perfect O benchmark, which is exactly why the source-safe regime matters: source separation exposes generalization difficulty that the supplied O split largely hides.

## What happened to the O-winning configuration?

C14 was the Best-O configuration. Under the S protocol, C14 was **retrained from scratch on S training data with the same frozen architecture/hyperparameters**; the O-trained weights were not reused.

That C14 configuration ranked third on S:

| Metric | C18 | C14 configuration on S |
|---|---:|---:|
| F1 | **0.819625** | 0.693642 |
| Precision | 0.823188 | 0.697674 |
| Recall | 0.816092 | 0.689655 |
| Accuracy | 0.837662 | 0.724675 |
| Balanced accuracy | 0.835771 | 0.721605 |
| ROC-AUC | 0.900236 | 0.793212 |
| PR-AUC | 0.899480 | 0.790797 |
| Parameters | 55,213 | 1,570,081 |

C18 is about 28.4 times smaller by parameter count and has an observed F1 advantage of 0.126 on this held-out environment.

## Uncertainty across held-out videos

Frame-level observations within a video are correlated, so uncertainty was estimated by **cluster bootstrap over video ID**, not by treating all 770 frames as independent.

Using 20,000 bootstrap resamples of the 29 held-out ENV-03 videos:

- C18 F1 95% percentile interval: **0.674 to 0.912**;
- C14-configuration F1 interval: **0.479 to 0.839**;
- paired C18 − C14 F1 difference: **+0.126**;
- paired 95% interval for that difference: **+0.019 to +0.261**.

These intervals are conditional on the observed selected candidates and are **not selection-adjusted** for choosing Best S from the same 20-candidate S test. They quantify source/video sampling uncertainty, not the full uncertainty of the model-selection procedure.

## Error structure

Best S made 125 errors: 61 false positives and 64 false negatives.

The errors are not spread uniformly across the 29 held-out videos. Major concentrations include:

- `notflip/0036`: 30/30 frames predicted positive;
- `notflip/0035`: 12 false positives;
- `flip/0039`: 11 false negatives;
- `notflip/0039`: 10 false positives;
- `flip/0041`: 10 false negatives;
- `flip/0035`: 10 false negatives.

Several held-out flip videos were classified perfectly, while a small number of videos contributed a large fraction of the total errors. This is consistent with meaningful source/video-level heterogeneity and reinforces why video-cluster uncertainty is more appropriate than naive frame-level uncertainty.

The per-video table and prioritized error cases are preserved in `results/scratch/S/`.

## Interpretation

S provides the first direct evidence in this rebuild that the scratch-CNN family can generalize beyond training-seen videos and beyond the training acquisition environments.

The result is **substantially weaker than O** but still clearly above the trivial majority baseline and stronger than the frozen baseline family previously evaluated for S. The gap between O and S supports the central methodological concern: the supplied Original split is much easier than a source-disjoint evaluation.

The correct conclusion is therefore not that the O result was invalid. Rather:

- O measures performance under the supplied, highly source-joint benchmark;
- S measures performance on an unseen acquisition environment;
- the two answer different questions;
- source-safe performance is meaningfully lower and more variable across videos.

No S candidate may now be modified based on S-test results.

## Next primary gate

Proceed to **T — Temporal-Safe** using the same fixed C01–C20 bank, the frozen T split, 20 epochs, no validation, and the same checkpoint-before-test gate. T asks whether the model generalizes across the frozen temporal boundary without changing the candidate family.
