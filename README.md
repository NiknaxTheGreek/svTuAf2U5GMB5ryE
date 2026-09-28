# MonReader

MonReader is an image-classification project for predicting the supplied `flip` / `notflip` label from a single frame. It is designed around leakage-safe evaluation, reproducible configuration, and controlled comparisons across data splits, model families, and visual ablations.

The executable foundation currently provides deterministic configuration, device selection, optimizer construction, testing, and a synthetic smoke flow. Scientific results will be added only after the corresponding data and modeling workflows have been executed and validated.

## Quick start

Official development uses Python 3.13.5.

```bash
python -m pip install -r requirements.txt
python run_experiment.py --config configs/base.yaml
pytest -q
```

The current CLI command runs a tiny deterministic synthetic smoke flow. It does not train on MonReader data and does not produce scientific results.
