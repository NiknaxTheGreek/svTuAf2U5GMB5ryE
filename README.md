# MonReader

MonReader is an image-classification project for predicting the supplied `flip` / `notflip` label from a single frame. The project is being rebuilt from a clean baseline with leakage-safe evaluation, reproducible configuration, and controlled comparisons across data splits, model families, and visual ablations.

The current repository state contains the executable project foundation only. Dataset ingestion, split construction, baselines, CNN tuning, generalization experiments, ResNet18 comparison, ablations, temporal/video analysis, and final W&B-backed reporting are added in later validated phases.

## Quick start

Official development uses Python 3.13.5.

```bash
python -m pip install -r requirements.txt
python run_experiment.py --config configs/base.yaml
pytest -q
```

The current CLI command runs a tiny deterministic synthetic smoke flow. It does not train on MonReader data and does not produce scientific results.
