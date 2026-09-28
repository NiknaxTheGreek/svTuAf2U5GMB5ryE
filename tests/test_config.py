from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import apply_overrides, load_config, to_runtime


def test_valid_config_loads() -> None:
    config = load_config(Path("configs/base.yaml"))
    runtime = to_runtime(config)
    assert runtime.project_name == "monreader"
    assert runtime.dataset_id == "DATA-001"
    assert runtime.split_id == "SPLIT-001"
    assert runtime.optimizer_name == "adam"
    assert runtime.seed == 42


def test_safe_overrides_are_applied() -> None:
    config = load_config("configs/base.yaml")
    changed = apply_overrides(config, seed=7, device="cpu")
    assert changed.seed == 7
    assert changed.device == "cpu"
    assert changed.dataset_id == config.dataset_id
    assert changed.split_id == config.split_id
    assert changed.optimizer == config.optimizer


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "project_name: monreader\nrun_name: bad\ndataset_id: DATA-001\nsplit_id: SPLIT-001\nseed: 1\ndevice: cpu\nmode: synthetic_smoke\nunknown: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_invalid_optimizer_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_optimizer.yaml"
    path.write_text(
        "project_name: monreader\nrun_name: bad\ndataset_id: DATA-001\nsplit_id: SPLIT-001\nseed: 1\ndevice: cpu\nmode: synthetic_smoke\noptimizer:\n  name: adagrad\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_invalid_override_is_rejected() -> None:
    config = load_config("configs/base.yaml")
    with pytest.raises(ValidationError):
        apply_overrides(config, seed=-1)
