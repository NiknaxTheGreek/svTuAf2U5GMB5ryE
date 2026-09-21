# Baseline

## Status

Step 5 baseline: COMPLETE.

The baseline was executed in GitHub Actions run #9 on the authoritative Google Drive dataset after the dataset audit, source-group reconstruction, and frozen leakage-safe split had all passed.

The frozen test partition was not used for feature extraction, fitting, threshold selection, or evaluation.

## Baseline definition

Primary baseline model:

- Input unit: one image/frame
- Feature representation: 26 simple global image-statistics features extracted from a 96 x 96 RGB resize
- Model: StandardScaler + LogisticRegression
- C: 1.0
- Solver: liblinear
- Random seed: 2026
- Positive class: flip
- Decision threshold: 0.5
- Validation threshold tuning: none
- Training frames: 1,795
- Validation frames: 597
- Test frames used: 0

The feature set is deliberately simple and non-deep-learning-based. It includes global brightness/contrast quantiles, dark/bright ratios, entropy, gradient/edge statistics, second-difference texture, RGB means/standard deviations, and coarse spatial brightness contrasts.

## Results

| Model/reference | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Logistic baseline — train | 0.8457 | 0.8692 | 0.8234 | 0.8540 |
| Logistic baseline — validation | 0.8370 | 0.7672 | 0.9207 | 0.8258 |
| Always flip — validation | 0.6539 | 0.4858 | 1.0000 | 0.4858 |
| Always notflip — validation | 0.0000 | 0.0000 | 0.0000 | 0.5142 |

Validation confusion matrix for the logistic baseline:

- True negatives: 226
- False positives: 81
- False negatives: 23
- True positives: 267

The locked baseline F1 to beat on the development validation partition is:

**0.8369905956112853**

The train-to-validation F1 difference is approximately 0.0087, so this simple model does not show a large frame-level generalization gap on the frozen source-group-disjoint split.

## Error concentration

Validation errors are strongly concentrated by source clip.

False positives:

- `notflip__0041`: 37 / 37 frames misclassified
- `notflip__0058`: 30 / 30 frames misclassified
- `notflip__0055`: 13 / 30 frames misclassified
- `notflip__0043`: 1 / 30 frames misclassified

Therefore 80 / 81 false positives come from three notflip source clips.

False negatives:

- `flip__0012`: 8
- `flip__0008`: 5
- `flip__0062`: 3
- `flip__0034`: 2
- `flip__0065`: 2
- `flip__0006`: 1
- `flip__0027`: 1
- `flip__0054`: 1

The error structure suggests that clip-specific visual conditions remain a major weakness of the simple global-statistics representation.

## Feature interpretation

The largest absolute standardized logistic coefficients include:

- red_mean: -3.4209
- red_std: -3.1112
- entropy_32bin: +2.5179
- edge_ratio: +2.4564
- gray_p10: -2.3240
- gray_std: -2.1122
- grad_y_mean: -2.0526
- gray_p25: -1.9918
- grad_mean: -1.8727
- green_std: -1.7298

These coefficients are descriptive rather than causal. Several image-statistics features are correlated, so individual coefficient magnitudes should not be interpreted as independent feature effects.

## Locked comparison rule

Future candidate image models must be compared on:

- the same single-image target,
- the same frozen source-group split,
- the same validation population,
- F1 for the positive `flip` class as the primary metric.

The test partition remains frozen until model architecture, preprocessing, and model-selection decisions are locked.

## Reproducibility

GitHub Actions run #9 completed successfully.

Artifact hashes:

- `image_stat_features_train_validation.csv`:
  `b493e334819292ea11a4696cb71a206658b69f8ffaf1a8f99b1d778b675d08eb`
- `validation_predictions.csv`:
  `9a4c7c2516d03ed00ba531698a8656e44b50224c89c182ab18ed0a28278e9153`
- `logistic_coefficients.csv`:
  `29d3e1be63b5861aa51a4a99506ef8dfbade422ce7de032db4b999cd221a9432`
