# Single-Image Challenger Model

## Status

Step 6 single-image model: COMPLETE.

The selected challenger is an ImageNet-pretrained **MobileNetV3-Small** classifier. It was trained and selected entirely with the frozen train/validation partitions from Step 4. The frozen test images were not opened or scored.

## Why this architecture

MobileNetV3-Small is appropriate to the project for two reasons:

1. It is a genuine convolutional image model capable of learning spatial page/hand/edge patterns that the Step 5 global-statistics baseline cannot represent.
2. It is designed for compact/mobile inference, which is aligned with the intended smartphone setting.

## Locked configuration

- Architecture: MobileNetV3-Small
- Pretraining: ImageNet `IMAGENET1K_V1`
- Input size: 160 x 160
- Positive class: `flip`
- Seed: 2026
- Batch size: 64
- Optimizer: AdamW
- Weight decay: 0.0001
- Decision threshold: 0.5
- Threshold tuning: none

Training schedule:

- Epochs 1-2: pretrained feature extractor frozen; replacement classifier head trained at learning rate 0.001.
- Epochs 3-8: final four feature blocks unfrozen; limited fine-tuning at learning rate 0.0002.
- Candidate checkpoint selection: highest validation F1.
- Selected epoch: **4**.

Training augmentation:

- RandomResizedCrop to 160 x 160, scale 0.90-1.00, aspect ratio 0.95-1.05
- Random horizontal flip, probability 0.5
- Brightness/contrast jitter, magnitude 0.10
- ImageNet normalization

Validation preprocessing:

- Deterministic resize to 160 x 160
- ImageNet normalization

## Validation result

| Model | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Step 5 logistic baseline | 0.83699 | 0.76724 | 0.92069 | 0.82580 |
| MobileNetV3-Small | **0.92953** | **0.90523** | **0.95517** | **0.92965** |

Absolute F1 improvement over the locked baseline:

**+0.09254**

Selected-checkpoint confusion matrix:

- True negatives: 278
- False positives: 29
- False negatives: 13
- True positives: 277

The challenger therefore satisfies the predeclared requirement to beat the baseline on the same validation population and primary metric.

## Epoch history

Validation F1 by epoch:

- Epoch 1, head only: 0.60143
- Epoch 2, head only: 0.83189
- Epoch 3, fine-tune: 0.91986
- Epoch 4, fine-tune: **0.92953** <- selected
- Epoch 5, fine-tune: 0.91269
- Epoch 6, fine-tune: 0.91909
- Epoch 7, fine-tune: 0.92632
- Epoch 8, fine-tune: 0.92256

Selection used validation F1 only. Epoch 4 is frozen from this point forward.

## Error analysis

The selected model makes 42 validation errors versus 104 for the Step 5 baseline.

Relative to the baseline:

- 71 baseline errors are corrected.
- 9 previously correct baseline frames become errors.
- Net error reduction: 62 frames.

Remaining errors by source group:

- `notflip__0058`: 18
- `notflip__0055`: 11
- `flip__0034`: 4
- `flip__0062`: 4
- `flip__0012`: 3
- `flip__0054`: 1
- `flip__0063`: 1

Notably, the baseline's 37-frame failure on `notflip__0041` is eliminated completely.

## Test information boundary

During Step 6:

- Train pixels opened: 1,795
- Validation pixels opened: 597
- Test rows present in metadata: 597
- Test pixels opened: **0**
- Test metrics computed: **false**

The test set remains an untouched final holdout.

## Selected checkpoint

Selected checkpoint SHA-256:

`49f2256715eb293adb2caaac410b6b4ed7dd6a2c3ba99828504f672a244170f1`

The exact checkpoint is retained in the GitHub Actions artifact from Step 6 run #1. Final test evaluation must use this exact selected checkpoint rather than selecting a new epoch after test access.

## Environment

The successful workflow pinned:

- Python 3.12
- PyTorch 2.8.0 CPU
- torchvision 0.23.0

GitHub Actions workflow: `.github/workflows/model-step6.yml`
Successful run ID: `35637600828`
