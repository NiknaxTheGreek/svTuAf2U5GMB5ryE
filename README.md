# Single-Image Page-Flip Detection

A reproducible computer-vision project for classifying whether a single smartphone frame shows a page being flipped.

The primary metric is **F1 for the `flip` class**. A downstream sequence extension aggregates frozen frame probabilities across source clips.

## Final reported result

The locked single-image model is **ImageNet-pretrained MobileNetV3-Small** using a fixed 0.5 threshold.

| Stage | Unit | F1 | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|
| Simple logistic baseline | validation frame | 0.83699 | 0.76724 | 0.92069 | 0.82580 |
| MobileNetV3-Small | validation frame | 0.92953 | 0.90523 | 0.95517 | 0.92965 |
| **MobileNetV3-Small final holdout** | **test frame** | **0.75145** | **0.85153** | **0.67241** | **0.78392** |
| Sequence mean aggregation | validation clip | 0.96296 | 0.92857 | 1.00000 | 0.95652 |
| Sequence mean aggregation | exploratory test clip | 0.80000 | 0.83333 | 0.76923 | 0.79167 |

The sequence result is a different unit of analysis and is not a replacement for the single-frame final result.

## Why the original folders were not used as train/test

The supplied archive contains 2,989 JPEG frames. Audit showed that all 115 source groups represented in the supplied `testing/` folder were also represented in `training/`. The two folders therefore partition frames from the same clips rather than providing clip-disjoint evaluation.

The project reconstructs canonical source groups as:

```
source_group = (label, VideoID)
```

and freezes a new source-group-disjoint split:

| Partition | Source groups | Frames |
|---|---:|---:|
| Train | 70 | 1,795 |
| Validation | 23 | 597 |
| Test | 24 | 597 |

Source-group overlap across partitions is zero.

## Methodology

1. Verify the supplied archive and image inventory.
2. Reconstruct source clips from `<VideoID>_<FrameNumber>.jpg`.
3. Audit cross-folder source overlap and duplicate risk.
4. Build a deterministic source-group-disjoint split.
5. Establish a simple image-statistics logistic baseline.
6. Train an ImageNet-pretrained MobileNetV3-Small challenger.
7. Select the checkpoint by validation F1 only.
8. Evaluate the exact selected checkpoint once on the frozen test holdout.
9. Perform post-holdout diagnostics without changing the reported model.
10. Aggregate immutable frame probabilities for an exploratory sequence extension.

## Important evaluation governance

The final test holdout is **consumed**.

The reported single-frame result remains:

- F1: **0.751445**
- Precision: **0.851528**
- Recall: **0.672414**
- Accuracy: **0.783920**
- TN / FP / FN / TP: **273 / 34 / 95 / 195**

No threshold tuning, epoch selection, or model changes were performed after test access.

## Key diagnostic finding

The validation-to-test F1 change is substantial:

```
0.92953 validation -> 0.75145 final test
```

Errors are strongly source-clip dependent. Five difficult flip clips account for 85 of the 95 final false negatives. Mild JPEG compression and blur have little effect, while +15% brightness and +15% contrast reduce F1 materially.

See:

- `docs/FINAL_REPORT.md`
- `docs/DATASET_AUDIT.md`
- `docs/SPLIT_POLICY.md`
- `docs/SINGLE_IMAGE_MODEL.md`
- `docs/FINAL_TEST_EVALUATION.md`
- `docs/POST_HOLDOUT_ANALYSIS.md`
- `docs/SEQUENCE_EXTENSION.md`

## Repository layout

```
.
├── audit/       # locked dataset/split audit summaries
├── docs/        # methodology and final report
├── results/     # committed compact metric/result summaries
├── scripts/     # executable audit, training, evaluation, diagnostics
├── splits/      # frozen source-group split
├── PROJECT_STATE.yaml
├── requirements.txt
└── requirements-torch-cpu.txt
```

The raw dataset and trained checkpoint are not committed to the repository.

## Environment

Verified clean-run environment:

- Python 3.12.14
- gdown 6.4.0
- NumPy 2.5.3
- pandas 3.0.6
- Pillow 12.3.0
- scikit-learn 1.9.1
- SciPy 1.18.1
- PyTorch 2.8.0 CPU
- torchvision 0.23.0 CPU

Install:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch-cpu.txt
```

## Reproducibility check

The repository contains a zero-test-access verifier:

```bash
python scripts/verify_repository.py
```

It checks the frozen split hash/counts, committed result locks, required documents, project-state consistency, and Python syntax without rerunning or reusing the consumed test holdout.

Historical GitHub Actions workflows preserve the authoritative data audit, model training, one-shot final evaluation, robustness analysis, and sequence extension runs.
