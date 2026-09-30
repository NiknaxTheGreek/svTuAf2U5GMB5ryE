# MonReader V2 Scratch-CNN Protocol

Status: **FROZEN BEFORE V2 SCRATCH-CNN TRAINING AND TEST EVALUATION**

## Candidate bank

Canonical file: `configs/MONREADER_FIXED_20_CONFIG_BANK.json`

SHA-256: `a4bebfc6c21df01e9294dacccb26ba84602cb3800b20414ad1d51f6f343c3b12`

Exactly the same C01–C20 configurations are used in O, S, T and ST.

## Architecture

For candidate depth `d` and `start_filters=f`, create `d` blocks:

1. `Conv2d(in, out, 3×3, stride=1, padding=1, bias=False)`;
2. `BatchNorm2d(out)`;
3. ReLU;
4. `MaxPool2d(2×2, stride=2)`.

The first block uses `out=f`; filters double after every block. The final feature map is reduced by adaptive global average pooling, followed by the candidate dropout and one linear output logit.

Initialization:
- convolution weights: Kaiming normal, fan-out, ReLU;
- output-linear weight: Kaiming normal, fan-in, linear;
- biases: zero;
- BatchNorm scale: one; BatchNorm bias: zero.

## Input preprocessing

- decode as RGB;
- aspect-ratio-preserving resize to fit a 224×398 (width×height) canvas;
- bilinear interpolation with antialiasing;
- centered constant-black padding if needed;
- cache the resulting tensor losslessly as uint8;
- convert to float and divide by 255 at model input;
- no augmentation in the primary candidate comparison.

For the verified DATA-001 images (1080×1920), the aspect ratio fits the primary portrait canvas without material padding.

## Training

- PyTorch 2.10.0 CPU reference execution;
- BCEWithLogitsLoss;
- natural class distribution;
- no class weights;
- no resampling;
- shuffled training batches;
- all training samples used each epoch (`drop_last=False`);
- deterministic seed = candidate seed (42 for all frozen candidates);
- exactly 20 epochs;
- no validation set;
- no early stopping;
- no learning-rate scheduler;
- no gradient clipping;
- final epoch-20 state is the frozen checkpoint.

## Optimizers

The frozen bank varies optimizer name, learning rate and weight decay. Optimizer arguments not represented in the bank are frozen here **before training** to the pinned PyTorch 2.10 defaults:

- Adam: betas=(0.9,0.999), eps=1e-8, amsgrad=False;
- AdamW: betas=(0.9,0.999), eps=1e-8, amsgrad=False;
- RMSprop: alpha=0.99, eps=1e-8, momentum=0.0, centered=False.

This resolves an implementation detail that the candidate JSON did not encode. Historical pre-V2 optimizer settings are non-authoritative.

## O execution gate

1. Build the O training cache from **only** the 2,392 frozen O training rows.
2. Train all C01–C20 for exactly 20 epochs without loading any O test or validation row.
3. Round-trip load every final checkpoint.
4. Verify all 20 checkpoint identities, configurations, epoch numbers and hashes.
5. Only after all 20 pass may the O test cache be built and the 597-image O test evaluated.
6. Each frozen candidate receives exactly one batch evaluation on the O test.
7. Rank by F1, then balanced accuracy, then PR-AUC, then lower trainable parameter count, then candidate ID.
8. The result is explicitly a **selection-benchmark estimate**.

No O-test result may change the candidate configuration or retrain a candidate.
