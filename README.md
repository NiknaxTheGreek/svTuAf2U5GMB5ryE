# MonReader

MonReader is an image-classification project for predicting the supplied `flip` / `notflip` label from a single frame. The scientific objective is reliable frame-level classification with credible generalization beyond the sources and capture conditions seen during fitting.

## Data

The authoritative archive contains 2,989 RGB JPEG frames: 1,452 `flip` and 1,537 `notflip`. The supplied benchmark contains 2,392 training frames and 597 testing frames. Every image is 1080 × 1920 pixels (aspect ratio 0.5625), and the decoded-pixel audit found no exact duplicate groups.

The filename structure provides a four-digit video identifier and `FrameNumber`. Because the numeric namespace is reused across class folders, the immutable manifest keeps an unambiguous class-namespaced video identity. A reviewed environment/session map is maintained separately from the raw naming metadata.

## Evaluation design

Five public-facing regimes are frozen before modeling expands: **Original**, **Random Stratified**, **Source-Safe**, **Temporal-Ordered**, and **Source-Safe + Temporal**. A separate video-disjoint split is retained to measure the effect of the stricter reviewed environment/session grouping instead of hiding that assumption.

Temporal ordering is defined by ascending `FrameNumber`. The primary Temporal-Ordered analysis uses only videos with at least 20 frames and assigns earliest 80% to training, middle 10% to validation, and latest 10% to testing. Source-Safe + Temporal uses a complete held-out environment/session for final testing and chronological fitting/validation inside the remaining eligible videos.

See `notebooks/01_data_audit.ipynb` for the data audit and `notebooks/02_split_design.ipynb` for split construction and leakage checks.

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
