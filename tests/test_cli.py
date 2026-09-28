import json
import subprocess
import sys


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
    assert payload["seed"] == 42


def test_cli_rejects_unknown_override() -> None:
    completed = subprocess.run(
        [sys.executable, "run_experiment.py", "--config", "configs/base.yaml", "--learning-rate", "0.5"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
