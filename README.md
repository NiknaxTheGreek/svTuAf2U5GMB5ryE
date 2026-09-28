# MonReader

MonReader is an image-classification project for predicting the supplied `flip` / `notflip` label from a single frame. The scientific objective is reliable frame-level classification with credible generalization beyond the videos and capture conditions seen during fitting.

## Data

| Images | `flip` | `notflip` | Supplied train | Supplied test | Canonical video groups | Native size | Aspect ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,989 | 1,452 | 1,537 | 2,392 | 597 | 117 | 1080 × 1920 | 0.5625 |

The first filename field is a four-digit raw video identifier and the second is `FrameNumber`. Because the numeric namespace is reused across class folders, the immutable manifest preserves the raw ID and uses a class-namespaced video group to avoid silently merging ambiguous sequences. Of the 117 canonical video groups, 115 occur in both supplied partitions, so the supplied split is retained as the **Original** benchmark rather than treated as unseen-video evidence.

All images decoded successfully during canonical ingestion, and the exact decoded-pixel check found no duplicate groups. Training-only RGB statistics, dataset lineage, and a small deterministic representative sample are committed with the manifest. See `notebooks/01_data_audit.ipynb` for the reviewer-facing data review.

## Workflow

The project proceeds from authoritative-data validation through leakage-safe split construction, simple baselines, scratch-CNN tuning, unseen-source and temporal comparisons, pretrained ResNet18 comparison, controlled visual ablations, and a final temporal/video extension. The public workflow uses the descriptive regimes **Original**, **Random Stratified**, **Source-Safe**, **Temporal-Ordered**, and **Source-Safe + Temporal**.

## Quick start

Official development uses Python 3.13.5.

```bash
python -m pip install -r requirements.txt
pytest -q
python run_experiment.py --config configs/base.yaml --device cpu
python -m scripts.execute_notebook notebooks/01_data_audit.ipynb --output /tmp/01_data_audit.executed.ipynb
```

The current experiment CLI is a deterministic synthetic smoke flow; it does not train on MonReader data or report scientific model performance.
