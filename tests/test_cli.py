import json
import subprocess
import sys
from pathlib import Path


def test_cli_synthetic_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, "run_experiment.py", "--config", "configs/base.yaml", "--device", "cpu"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "ok"
    assert payload["mode"] == "synthetic_smoke"
    assert payload["device"] == "cpu"
    assert payload["dataset_id"] == "DATA-001"
    assert payload["split_id"] == "SPLIT-001"
    assert payload["seed"] == 42


def test_cli_rejects_unknown_override() -> None:
    completed = subprocess.run(
        [sys.executable, "run_experiment.py", "--config", "configs/base.yaml", "--learning-rate", "0.5"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_cli_rejects_incompatible_split(tmp_path: Path) -> None:
    config = tmp_path / "bad_split.yaml"
    config.write_text(
        "project_name: monreader\nrun_name: bad-split\ndataset_id: DATA-001\nsplit_id: SPLIT-999\nseed: 42\ndevice: cpu\nmode: synthetic_smoke\noptimizer:\n  name: adam\n  learning_rate: 0.001\n  weight_decay: 0.0\n  momentum: 0.9\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "run_experiment.py", "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "Incompatible dataset/split pair" in completed.stderr
