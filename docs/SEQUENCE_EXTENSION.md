# Sequence-Level Extension

## Status

Step 9 sequence extension: COMPLETE.

This stage addresses the downstream challenge of determining whether a sequence contains page-flipping activity by aggregating the already-frozen single-frame probabilities from the Step 6 MobileNetV3-Small model.

This does **not** replace the primary single-image result. The final single-frame holdout F1 remains 0.75145.

## Governance

The Step 7 test holdout had already been consumed before this extension was created. Therefore:

- the frame classifier is unchanged;
- the frame decision threshold remains 0.5;
- no image model retraining occurs;
- the sequence aggregation rule is selected using validation probabilities only;
- test probabilities are not used to choose the aggregation method;
- sequence test performance is reported as **post-holdout exploratory**.

## Candidate aggregation rules

Seven simple sequence aggregators were predeclared:

1. mean frame probability
2. median frame probability
3. fraction of frames with probability >= 0.5
4. 75th-percentile probability
5. 90th-percentile probability
6. mean of the top 3 frame probabilities
7. maximum frame probability

Each produces one sequence score, which is classified at the fixed threshold 0.5.

Tie-break rule:

**highest validation F1, then fixed candidate priority**.

## Validation selection

Validation contains 23 independent source groups.

| Aggregator | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Mean probability | **0.96296** | 0.92857 | 1.00000 | 0.95652 |
| Median probability | **0.96296** | 0.92857 | 1.00000 | 0.95652 |
| Positive-frame fraction | **0.96296** | 0.92857 | 1.00000 | 0.95652 |
| 75th percentile | 0.92857 | 0.86667 | 1.00000 | 0.91304 |
| 90th percentile | 0.92857 | 0.86667 | 1.00000 | 0.91304 |
| Top-3 mean | 0.92857 | 0.86667 | 1.00000 | 0.91304 |
| Maximum probability | 0.92857 | 0.86667 | 1.00000 | 0.91304 |

The three leading methods tie. The fixed tie-break therefore selects **mean probability**.

The only validation sequence error is:

- `notflip__0058`, mean flip probability 0.5054.

## Locked sequence rule

For a source sequence with frame probabilities (p_1, ..., p_n):

`sequence_score = mean(p_i)`

Prediction:

- flip if sequence_score >= 0.5
- notflip otherwise

No sequence-specific threshold tuning is performed.

## Exploratory test result

Applied to the already-consumed 24-group test set:

| Metric | Sequence result |
|---|---:|
| **F1** | **0.80000** |
| Precision | 0.83333 |
| Recall | 0.76923 |
| Accuracy | 0.79167 |

Confusion matrix:

- True negatives: 9
- False positives: 2
- False negatives: 3
- True positives: 10

Exploratory test sequence errors:

- `flip__0038`
- `flip__0041`
- `flip__0042`
- `notflip__0021`
- `notflip__0053`

The three false-negative flip sequences are the same persistent difficult clips identified in Step 8. This is consistent with aggregation reducing isolated frame noise but not rescuing clips where the frame model is systematically biased throughout much of the sequence.

## Interpretation

Mean aggregation converts noisy frame-level decisions into a single clip-level score and performs strongly on validation. On the exploratory test groups it achieves F1 0.80, but this number is **not directly comparable** to the single-frame F1 0.75145 because the unit of analysis is different: 24 sequences versus 597 frames.

The sequence result supports the product-level idea that temporal aggregation can stabilize frame predictions. It does not eliminate source-domain failures where most frames of an entire clip are assigned the wrong side of the decision boundary.

## Reproducibility

The workflow uses immutable prediction artifacts only:

- Step 6 validation predictions SHA-256:
  `e3521f7cf87dd7c1ad8537ab869a452cae62dde7cac785556c3f4a4f1f270d92`
- Step 7 final-test predictions SHA-256:
  `3db3083ddccb0fafe9142f1e183104b3b193311033fcb231c3ba6ca9211822cf`

Successful workflow run: `35776906695`.

Sequence outputs:

- `sequence_validation_candidates.csv` SHA-256:
  `c72ac31d8803c486932056a1606a9500eeb0fd50cd0ac647ebd8910ceb94425b`
- `sequence_test_predictions.csv` SHA-256:
  `c0c7e24fa44c30b915ac66b6e05b7b5ef346ba0e59c7f64d349f352fd260d607`
