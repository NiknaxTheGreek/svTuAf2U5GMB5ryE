# MonReader

MonReader is an image-classification project for predicting the supplied `flip` / `notflip` label from a single frame. The scientific objective is reliable frame-level classification with credible generalization beyond the sources and capture conditions seen during fitting.

## Data

The authoritative archive contains 2,989 RGB JPEG frames: 1,452 `flip` and 1,537 `notflip`. The supplied benchmark contains 2,392 training frames and 597 testing frames. Every image is 1080 × 1920 pixels (aspect ratio 0.5625).

The filename structure provides a four-digit video identifier and `FrameNumber`. Because the numeric video namespace is reused across class folders, the immutable manifest keeps both the raw numeric ID and an unambiguous class-namespaced video ID. Broader source/environment grouping is reviewed separately before source-safe evaluation. The Phase-2 decoded-pixel audit found no exact duplicate groups.

The canonical original-RGB dataset is registered as `DATA-001`; its manifest, lineage metadata, training-only RGB statistics, and a small representative documentation sample are committed. See `notebooks/01_data_audit.ipynb` for the reviewer-facing audit.

## Workflow

The project proceeds from authoritative-data validation through leakage-safe split construction, simple baselines, scratch-CNN tuning, unseen-source and temporal comparisons, pretrained ResNet18 comparison, controlled visual ablations, and a final temporal/video extension. Experiment tracking is validated with W&B only after the core methodology is stable.

## Quick start

Official development uses Python 3.13.5.

```bash
python -m pip install -r requirements.txt
python run_experiment.py --config configs/base.yaml
pytest -q
```

The current CLI command is a deterministic synthetic smoke flow; it does not train on MonReader data or report scientific model performance.
