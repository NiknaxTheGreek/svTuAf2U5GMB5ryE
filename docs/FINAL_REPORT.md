# Final Technical Report

## 1. Objective

The primary task is binary computer-vision classification from a **single image**:

- positive class: `flip`
- negative class: `notflip`
- primary metric: F1 for `flip`

The downstream product challenge is sequence-level detection of page-flipping activity. That extension is treated separately from the primary single-frame result.

## 2. Dataset audit

The authoritative archive contains **2,989** readable 1080 x 1920 RGB JPEG frames:

- flip: 1,452
- notflip: 1,537

The supplied folder counts are:

- training / flip: 1,162
- training / notflip: 1,230
- testing / flip: 290
- testing / notflip: 307

Archive SHA-256:

`033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f`

No unreadable images or exact cross-split SHA-256 duplicates were found.

### Critical split finding

Filenames follow `<VideoID>_<FrameNumber>.jpg`. Reconstructing clips by `(label, VideoID)` produced 117 canonical source groups:

- 65 flip groups
- 52 notflip groups

All **115** source groups represented in the supplied `testing/` folder are also represented in `training/`. Frame numbering and perceptual similarity further support that these folders partition frames from the same source clips.

Therefore the supplied folders were **not** treated as an independent evaluation split.

## 3. Leakage-safe evaluation design

A deterministic source-group-disjoint split was created using only:

- class label,
- canonical source-group ID,
- frame count.

No image pixels, model predictions, or downstream evaluation results were used to select the split.

| Partition | Groups | Frames | Flip frames | Notflip frames |
|---|---:|---:|---:|---:|
| Train | 70 | 1,795 | 872 | 923 |
| Validation | 23 | 597 | 290 | 307 |
| Test | 24 | 597 | 290 | 307 |

Source-group overlap is zero for train/validation, train/test, and validation/test.

The test set was frozen before model development.

## 4. Baseline

A deliberately simple baseline used 26 global image-statistics features extracted from a 96 x 96 resize, followed by:

`StandardScaler -> LogisticRegression`

Threshold was fixed at 0.5.

Validation result:

- F1: 0.83699
- Precision: 0.76724
- Recall: 0.92069
- Accuracy: 0.82580

This baseline was used only as the locked development comparison. It was not evaluated on the final frozen test holdout.

## 5. Single-image model

The challenger is **MobileNetV3-Small** with ImageNet `IMAGENET1K_V1` weights.

Configuration:

- input: 160 x 160 RGB
- batch size: 64
- optimizer: AdamW
- threshold: 0.5
- epochs 1-2: classifier head only
- epochs 3-8: final four feature blocks unfrozen
- selected checkpoint: epoch 4
- model selection criterion: validation F1 only

Selected checkpoint SHA-256:

`49f2256715eb293adb2caaac410b6b4ed7dd6a2c3ba99828504f672a244170f1`

Validation result:

- F1: **0.92953**
- Precision: 0.90523
- Recall: 0.95517
- Accuracy: 0.92965

Absolute validation F1 improvement over the baseline:

**+0.09254**

## 6. Final one-shot holdout evaluation

Before test access, the exact checkpoint hash, epoch, threshold, 24 test source groups, and 597 file paths were verified without opening test pixels.

The selected epoch-4 checkpoint was then evaluated once on the frozen test holdout.

### Final reported single-frame result

| Metric | Result |
|---|---:|
| **F1** | **0.75145** |
| Precision | 0.85153 |
| Recall | 0.67241 |
| Accuracy | 0.78392 |
| True negatives | 273 |
| False positives | 34 |
| False negatives | 95 |
| True positives | 195 |

The validation-to-test F1 change is **-0.17809**.

No test-based threshold tuning, retraining, epoch reselection, or model replacement was performed.

## 7. Error and robustness analysis

The final holdout contains 129 incorrect frames:

- 95 false negatives
- 34 false positives

The errors are concentrated by source clip rather than uniformly distributed. Five flip clips account for **85 / 95** false negatives.

A 20,000-replicate stratified source-group bootstrap gives an approximate 95% interval for F1 of:

**0.614 to 0.883**

This wide interval reflects strong between-clip variation and the small number of independent test clips.

### Confidence

Mean probability margin from the decision boundary:

- correct frames: 0.3898
- incorrect frames: 0.2612

Frames within 0.05 of the threshold have a 60% observed error rate, while the highest-confidence 0.40-0.50 margin bin has an 8.8% error rate.

### Perturbation diagnostics

Using the frozen model and threshold:

| Condition | F1 | Change vs original |
|---|---:|---:|
| Original | 0.75145 | — |
| Brightness +15% | 0.69636 | -0.05509 |
| Contrast +15% | 0.70281 | -0.04863 |
| JPEG quality 70 | 0.74319 | -0.00825 |
| Gaussian blur r=1 | 0.74903 | -0.00241 |
| Contrast -15% | 0.75049 | -0.00095 |
| Brightness -15% | 0.75836 | +0.00692 |

The model is more sensitive to brighter/higher-contrast inputs than to mild blur or JPEG recompression.

These are post-holdout diagnostics and were not used to alter the reported model.

## 8. Sequence extension

Seven simple clip aggregators were compared on the **validation** source groups using immutable frame probabilities.

Mean probability, median probability, and positive-frame fraction tied at validation F1 = 0.96296. A fixed priority rule selected **mean probability**.

Sequence rule:

`sequence_score = mean(frame flip probabilities)`

with threshold 0.5.

Validation sequence performance over 23 clips:

- F1: 0.96296
- Precision: 0.92857
- Recall: 1.00000
- Accuracy: 0.95652

Applied afterward to the already-consumed 24 test clips, the exploratory sequence result is:

- F1: 0.80000
- Precision: 0.83333
- Recall: 0.76923
- Accuracy: 0.79167

This sequence result is **not directly comparable** with the single-frame final F1 because the unit of analysis differs.

## 9. Limitations

1. Labels appear to be clip-derived and inherited by frames; independent frame-level adjudication is not verified.
2. There are only 117 independent canonical source groups, so source-level uncertainty is substantial.
3. Model selection produced strong validation performance but a clear source-domain generalization gap on the final holdout.
4. The final holdout is consumed; any subsequent model tuning requires a new untouched evaluation protocol to support a new final claim.
5. Sequence evaluation is post-holdout exploratory rather than a fresh independent final evaluation.
6. Robustness tests cover only a small fixed set of mild synthetic perturbations.

## 10. Reproducibility

The repository pins the verified environment and records:

- authoritative dataset SHA-256,
- frozen split manifest and SHA-256,
- selected checkpoint SHA-256,
- compact locked metric files,
- workflow run IDs/artifact IDs,
- deterministic analysis scripts.

The raw dataset and model checkpoint are intentionally not committed. Historical GitHub Actions runs provide execution receipts for the authoritative audit, model training, final one-shot evaluation, robustness analysis, and sequence extension.

Run the repository consistency verifier with:

```bash
python scripts/verify_repository.py
```

This verifies the committed state without re-consuming the final test holdout.

## 11. Conclusion

The project demonstrates why source-aware validation matters for frame datasets derived from video. A naive use of the supplied folders would mix frames from the same clips across development and evaluation.

After enforcing clip-level separation, MobileNetV3-Small improved validation F1 from 0.83699 to 0.92953, but achieved **0.75145 F1 on the untouched final frame-level holdout**, revealing a meaningful source-domain generalization gap.

The sequence extension shows that simple aggregation can stabilize predictions at clip level, but persistent source-specific failures remain the central modeling limitation.
