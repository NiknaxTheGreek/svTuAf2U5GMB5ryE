from __future__ import annotations

import json
from pathlib import Path

from scripts.record_frozen_config_failure import main as failure_main
from scripts.run_scratch_config import load_frozen_config


def test_generic_failure_recorder_module_imports() -> None:
    assert callable(failure_main)


def test_generic_cached_runner_uses_same_frozen_config_schema(tmp_path: Path) -> None:
    config = {
        "trial_id": "stage2-001",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "depth": 2,
        "start_filters": 16,
        "seed": 42,
        "parameter_count": 5169,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_frozen_config(path)
    assert loaded["parameter_count"] == 5169
