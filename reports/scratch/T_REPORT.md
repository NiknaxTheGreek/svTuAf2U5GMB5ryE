# MonReader V2 — Temporal-Safe (T) Scratch-CNN Selection Benchmark

## Question

How well does the frozen C01–C20 scratch-CNN candidate family generalize from **earlier observed frames to later observed frames of the same videos**?

T is a temporal-shift benchmark, not a source-disjoint benchmark. All 117 canonical videos contribute earlier frames to training and later frames to test. The 624-image T test is therefore deliberately source-joint but temporally separated.

Because the T test selects among 20 already-frozen candidates, the winning result is a **selection-benchmark estimate**.

## Execution integrity

- training rows: 2,365;
- test rows: 624;
- videos: 117;
- train/test video overlap: 117 by design;
- temporal boundary: earlier frames train, later frames test;
- threshold: 0.5;
- exactly 20 epochs;
- no validation;
- no early stopping;
- all 20 candidate jobs succeeded;
- all 20 final epoch-20 checkpoints were hash-verified before T test construction;
- every candidate received exactly one T-test evaluation.

The checkpoint registry reported `PASS_ALL_20_FROZEN_BEFORE_T_TEST` with `test_evaluation_opened=false`.

The evidence artifact SHA256 was independently verified as `7301e4c7d28e7296684acb61930d6f7a652eeac009fb915be8ecbb648b010ea6`.

A copied workflow job label still read `evaluate-s-once`; this was cosmetic. The executed verifier/evaluator modules, cache paths, split hash, checkpoint registry, output directory and result files were all T-specific.

## Best T

**C06** is Best T.

| Metric | C06 |
|---|---:|
| F1 | **0.880000** |
| Precision | 0.969349 |
| Recall | 0.805732 |
| Accuracy | 0.889423 |
| Balanced accuracy | 0.889963 |
| ROC-AUC | 0.951757 |
| PR-AUC | 0.963690 |
| TN / FP / FN / TP | 302 / 8 / 61 / 253 |

C06 has 345,826 trainable parameters and used RMSprop, depth 5, start filters 15, dropout 0.0900, learning rate 0.0005904, weight decay 0.0001660 and batch size 34.

## Cross-regime configuration behavior

The Best-O configuration, C14, was retrained from scratch under T and ranked 20th by fixed-threshold F1:

- F1 = 0.2981;
- precision = 1.0000;
- recall = 0.1752;
- ROC-AUC = 0.9414;
- PR-AUC = 0.9567.

This combination is important. C14 still ranks positives above negatives very well, but at threshold 0.5 it predicts far too few positives. The temporal regime therefore exposes a substantial **probability calibration / operating-threshold shift** for this configuration. Under the frozen protocol the threshold remains 0.5; no threshold tuning is allowed after T-test observation.

The Best-S configuration C18, also retrained under T, ranked 18th at F1 0.5048.

## Uncertainty across videos

A 20,000-draw percentile cluster bootstrap resampling the 117 videos gives:

- C06 F1 95% interval: **0.837–0.918**;
- C14-configuration F1 interval: **0.186–0.408**;
- observed C06 − C14 F1 difference: **+0.582**;
- paired 95% interval: **+0.480 to +0.687**.

The intervals are not selection-adjusted for choosing Best T among 20 candidates.

## Error structure

C06 made 69 errors: 8 false positives and 61 false negatives. Errors are asymmetric because C06 is conservative at the fixed threshold.

Concentrated examples include:

- `notflip/0015`: 6 false positives;
- `flip/0062`: 5 false negatives;
- `flip/0010`: 4 false negatives;
- `flip/0060`: 4 false negatives;
- `notflip/0042`: 2 false positives.

Most not-flip videos are classified cleanly, while several flip videos contribute the majority of misses.

## Interpretation

T shows that the scratch-CNN family retains strong performance when asked to predict later frames, but temporal shift materially changes which configuration is best.

This result should not be interpreted as unseen-user or unseen-environment performance because the same videos appear on both sides of T. It answers a narrower question: whether image-level flip/not-flip discrimination survives a strict chronological split within videos.

The sharp contrast between C06 and the O/S champions also shows that hyperparameter/model-selection behavior is regime-dependent.

No T candidate may now be modified based on T-test results.

## Next primary gate

Proceed to **ST — Source + Temporal Safe**. ST combines the two constraints: training excludes ENV-03 and uses only earlier/context-safe material, while the primary test consists of later frames from the held-out source. This is the most stringent primary regime.
