# MonReader V2 — Fixed ResNet18 Comparator Protocol

## Status

Prospectively frozen before any ResNet18 test evaluation.

Config: `configs/RESNET18_FIXED_COMPARATOR_V2.json`  
SHA256: `82af40ff6071949ee91d8934341b42bbdd7391738e0cde23d8ad9b9084338cc8`

## Why this freeze exists

The V2 authority explicitly replaces the old adaptive ResNet18 search with **one fixed non-adaptive comparator recipe**, but it does not specify all numeric training hyperparameters. To avoid silently reviving validation/Bayesian selection, this protocol freezes one conventional transfer-learning recipe prospectively, before any ResNet18 test is opened.

No value below is derived from O/S/T/ST test performance.

## Regimes

Run the same fixed recipe independently on O, S, T and ST.

Each model trains only on that regime's frozen training population. The four final epoch-20 checkpoints must all exist and pass identity/hash checks before any ResNet18 test evaluation begins.

## Model

- torchvision ResNet18;
- exact weights enum: `ResNet18_Weights.IMAGENET1K_V1`;
- replace the original FC layer with Dropout(0.2) + Linear(512, 1);
- BCEWithLogitsLoss;
- threshold 0.5.

## Preprocessing

Use the ImageNet-1K V1 ResNet18 evaluation geometry:

- resize shorter side to 256;
- bilinear interpolation with antialiasing;
- center crop 224×224;
- scale to [0,1];
- normalize with mean [0.485, 0.456, 0.406];
- normalize with std [0.229, 0.224, 0.225];
- no augmentation.

## Training

- seed 42;
- natural class distribution;
- batch size 32;
- AdamW;
- shared weight decay 1e-4;
- epochs 1–5: freeze backbone; train only the replacement head at LR 1e-3;
- after epoch 5: unfreeze the entire model and recreate AdamW at LR 1e-4;
- epochs 6–20: full-network fine-tuning;
- no validation loader;
- no early stopping;
- no scheduler;
- no gradient clipping;
- final epoch-20 checkpoint only.

## Evaluation

- one test evaluation per regime;
- evaluate only after all four O/S/T/ST epoch-20 checkpoints are frozen;
- threshold remains 0.5;
- report F1, precision, recall, accuracy, balanced accuracy, ROC-AUC, PR-AUC and confusion matrix;
- preserve predictions and checkpoint hashes;
- report separately from scratch champion selection.

The ResNet18 results are a fixed pretrained-family comparator. They cannot replace Best O/S/T/ST scratch champions.

## Change control

Once the first ResNet18 test is opened, this recipe cannot be changed based on observed test performance. Any later alternate ResNet settings are exploratory and must be separately labeled.
