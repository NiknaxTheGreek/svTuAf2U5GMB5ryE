# MonReader V2 — Source + Temporal Safe (ST) Primary Benchmark

## Primary question

How well does the fixed C01–C20 scratch-CNN family generalize when both source and temporal constraints are applied simultaneously?

ST training uses 1,756 earlier frames from known environments ENV-01, ENV-02 and ENV-04. The primary test contains 161 later frames from the completely held-out environment ENV-03. Train/test canonical-video overlap is zero.

## Execution integrity

All 20 candidates trained for exactly 20 epochs with no validation and no test access. The checkpoint gate reported `PASS_ALL_20_FROZEN_BEFORE_ST_TEST` before the 161-image primary test was constructed. Every candidate was then evaluated exactly once at threshold 0.5.

The authoritative run is GitHub Actions run `36815939762`. Its evidence artifact independently matches SHA256 `f2a80a6be8c66127f7f17f730da32cd792529fdca4185dafebc6cf92136485c5`.

An earlier ST attempt failed at Python compilation before cache construction, model training, or test access and is not used scientifically.

## Best ST

**C07** is the frozen Best-ST candidate.

| Metric | C07 |
|---|---:|
| F1 | **0.704000** |
| Precision | 0.897959 |
| Recall | 0.578947 |
| Accuracy | 0.770186 |
| Balanced accuracy | 0.760062 |
| ROC-AUC | 0.908204 |
| PR-AUC | 0.915007 |
| TN / FP / FN / TP | 80 / 5 / 32 / 44 |

C07 has 259,936 parameters. Its frozen checkpoint SHA256 is `9886dbad7f73984a8333bee22da568be4a451b650dad6ee28a2713853b445b67`.

## Uncertainty

The required 5,000-resample bootstrap was applied at both frame and canonical-video levels with fixed seed 20261001.

C07 F1:

- frame bootstrap 95% interval: **0.609–0.790**;
- video-cluster bootstrap 95% interval: **0.465–0.859**.

The substantially wider video-cluster interval reflects strong between-video heterogeneity and the small 29-video held-out test population. These intervals are selection-conditioned and do not correct for selecting C07 from 20 candidates on the same ST test.

## Previously winning configurations under ST

The prior regime champions were retrained from scratch under the ST training population using their fixed configurations:

- Best-O configuration C14: F1 0.4646, rank 12;
- Best-S configuration C18: F1 0.1905, rank 17;
- Best-T configuration C06: F1 0.6055, rank 5.

C07 exceeds C14 by observed F1 +0.239; the paired video-cluster 95% interval is +0.072 to +0.405. C07 exceeds C18 by +0.514 with interval +0.267 to +0.702. Against C06 the observed difference is +0.098, but the paired interval spans zero (-0.028 to +0.251), so this test does not establish a clear C07-vs-C06 difference across the held-out videos.

## Error structure

C07 makes 37 errors: 5 false positives and 32 false negatives.

The most obvious video-level failures are concentrated: several held-out flip videos have all later frames missed, while most not-flip videos are handled cleanly. For example, `flip/0034`, `flip/0037`, `flip/0039`, `flip/0041` and `flip/0048` have zero true positives in their ST late-frame subsets, while `notflip/0036` contributes all five false positives.

This heterogeneity is why the video-cluster interval is the more conservative uncertainty summary.

## Scientific interpretation

ST is the strictest primary benchmark in the V2 design. It combines an unseen acquisition environment with a later-frame test. Its F1 is lower than O, S and T, but raw F1 values across regimes answer different questions and must not be treated as a global ranking.

Best ST is now frozen. No context diagnostic, error analysis, ablation, augmentation experiment, pretrained comparator, or later post-hoc analysis may replace C07 or alter this primary result.

## Remaining ST task

The split contains 1,072 context-only images that were intentionally excluded from primary scoring:

- 609 earlier ENV-03 frames (`context_unseen_early`);
- 463 later known-source frames (`context_known_future`).

The execution plan requires descriptive evaluation of frozen C07 on those two populations after primary selection. Those results are explanatory only and cannot change Best ST.
