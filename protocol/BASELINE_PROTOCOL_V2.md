# MonReader V2 Baseline Protocol

Status: **FROZEN BEFORE V2 BASELINE TEST EVALUATION**

This protocol defines the non-CNN baselines. It must not be redesigned after the baseline test results are inspected.

## Evaluation populations

The frozen O/S/T/ST manifests in `manifests/splits/` define train/test membership. Context-only ST rows are not used for baseline training or primary testing.

Positive class: **flip**.

Classification threshold for logistic regression: **0.5**.

## Majority-class baseline

For each regime independently:

1. count labels in that regime's training role only;
2. predict the training majority class for every primary-test sample;
3. if an exact training tie ever occurs, resolve deterministically to `notflip`.

Report F1, precision, recall, accuracy, balanced accuracy and confusion matrix. ROC-AUC and PR-AUC are not reported for this class-only baseline because it does not produce a continuous discrimination score.

## Handcrafted logistic-regression baseline

### Deterministic image preprocessing

Each source JPEG is decoded as RGB.

Images are resized with preserved aspect ratio to fit a **224 × 398 (width × height)** portrait canvas using bilinear interpolation, centered, and zero-padded if needed. Values are scaled to `[0,1]`.

This preprocessing is baseline-specific but intentionally close to the scratch-CNN base geometry. No augmentation is used.

### Feature Set A — simple grayscale/edge statistics

1. brightness = mean grayscale intensity;
2. contrast = grayscale standard deviation;
3. sharpness = variance of the Laplacian response;
4. entropy = Shannon entropy of a fixed 64-bin grayscale histogram;
5. edge density = fraction of pixels flagged by Canny edges with `sigma=1.0`.

### Feature Set B — Set A plus colour/intensity statistics

Set A plus:

- RGB channel means (3);
- RGB channel standard deviations (3);
- HSV hue mean and standard deviation;
- HSV saturation mean and standard deviation;
- grayscale median;
- grayscale 10th, 25th, 75th and 90th percentiles.

### Feature Set C — Set B plus structure/texture

Set B plus:

- normalized 16-bin grayscale intensity histogram;
- normalized uniform local-binary-pattern histogram with `P=8`, `R=1` (10 bins);
- normalized 9-bin gradient-orientation histogram over `[0, π)`, weighted by Sobel gradient magnitude.

No feature family is selected after looking at the tests. **A, B and C are all reported.**

### Logistic-regression fit

For each regime and each feature set:

- `StandardScaler` fit on that regime's training rows only;
- logistic regression:
  - L2 penalty;
  - `C=1.0`;
  - solver = `liblinear`;
  - `class_weight=None`;
  - `max_iter=2000`;
  - `random_state=42`;
- no resampling;
- no class weighting;
- no validation split;
- no feature selection;
- no hyperparameter tuning.

The fitted scaler is applied to that regime's test rows. The test set never contributes to scaling statistics or fitting.

## Metrics

For logistic regression report:

- F1;
- precision;
- recall;
- accuracy;
- balanced accuracy;
- ROC-AUC;
- PR-AUC / average precision;
- confusion matrix.

Predictions and fitted scaler/model coefficients are saved for auditability.

## Interpretation

These baselines answer whether simple image statistics already contain useful class signal. They do not replace the scratch CNN and they are not candidates in the 20-config scratch-CNN champion selection.
