# Final Test Evaluation

## Status

Step 7 final test evaluation: COMPLETE.

The frozen epoch-4 MobileNetV3-Small checkpoint selected in Step 6 was evaluated exactly once on the untouched 597-frame test holdout.

No training, epoch selection, threshold selection, or hyperparameter changes were performed using the test set.

## Preflight

Before any test image pixels were opened, GitHub Actions run `35645087781` verified:

- authoritative archive SHA-256:
  `033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f`
- selected checkpoint SHA-256:
  `49f2256715eb293adb2caaac410b6b4ed7dd6a2c3ba99828504f672a244170f1`
- selected epoch: 4
- decision threshold: 0.5
- frozen test frames: 597
- frozen test source groups: 24
- test pixels opened during preflight: 0
- test metrics computed during preflight: false

## Final one-shot test result

GitHub Actions run `35645274486` scored the frozen test holdout once.

| Metric | Final test result |
|---|---:|
| **F1** | **0.75145** |
| Precision | 0.85153 |
| Recall | 0.67241 |
| Accuracy | 0.78392 |

Confusion matrix:

- True negatives: 273
- False positives: 34
- False negatives: 95
- True positives: 195

Support:

- Flip: 290
- Notflip: 307
- Total: 597

## Validation-to-test generalization gap

The selected checkpoint achieved:

- Validation F1: 0.92953
- Final test F1: 0.75145
- Absolute change: -0.17809

This is a substantial source-group generalization gap. Because the test set is now consumed, this result is retained as-is rather than used to retune the model.

## Error structure

The final test contains 129 incorrect frames:

- 95 false negatives
- 34 false positives

Errors occur in 13 of the 24 test source groups; 11 groups have zero errors.

The false-negative burden is strongly source-group concentrated:

- `flip__0038`: 32 / 46 frames incorrect
- `flip__0041`: 18 / 20
- `flip__0042`: 16 / 30
- `flip__0047`: 11 / 25
- `flip__0045`: 8 / 24

These five flip groups account for 85 / 95 false negatives.

False positives are also concentrated:

- `notflip__0053`: 23 / 30 frames incorrect

That single source group accounts for 23 / 34 false positives.

The concentration indicates that performance varies materially across source clips. This supports further post-holdout diagnostic work on visual/domain variation, but the final reported model and final test metrics must not be changed in response.

## Holdout governance

The test holdout is now **CONSUMED**.

From this point forward:

- the final test metrics above are immutable for this model-development cycle;
- the epoch-4 checkpoint remains the reported selected model;
- threshold 0.5 remains the reported threshold;
- no post-test tuning may be presented as part of the same untouched-holdout evaluation;
- any future model changes require a new evaluation protocol/new untouched holdout or must be clearly labelled exploratory/post-holdout.

## Reproducibility receipts

Final result artifact hashes:

- `test_predictions.csv`:
  `3db3083ddccb0fafe9142f1e183104b3b193311033fcb231c3ba6ca9211822cf`
- `test_errors_by_group.csv`:
  `36411ad60a5418a346ef650b631f0c202b580dea5d2ed4b436f44a63b4f281f5`
- `test_manifest.csv`:
  `82724e9cb5939a4eeea7ae961a2b8cb40f4e3c8dc127680f9dca82b1f82bf99f`

Final-test artifact ID: `10659537469`.
