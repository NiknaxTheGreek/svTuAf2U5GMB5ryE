from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_scratch_config import load_frozen_config


def _config(trial_id: str = "stage2-001") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "depth": 2,
        "start_filters": 16,
        "seed": 42,
        "parameter_count": 5073,
    }


def test_generic_frozen_config_validation(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_frozen_config(path)
    assert loaded["trial_id"] == "stage2-001"


def test_stage2_aggregate_requires_all_30(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    for index in range(1, 30):
        (root / f"stage2-{index:03d}.json").write_text(
            json.dumps({"trial_id": f"stage2-{index:03d}", "status": "failed"}),
            encoding="utf-8",
        )
    output = tmp_path / "aggregate.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.aggregate_scratch_stage2",
            "--results-root",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0


def test_canonical_selection_uses_source_gate_f1(tmp_path: Path) -> None:
    top5 = {
        "candidate_ids": [f"candidate-{index}" for index in range(1, 6)]
    }
    top5_path = tmp_path / "top5.json"
    top5_path.write_text(json.dumps(top5), encoding="utf-8")
    root = tmp_path / "gate"
    root.mkdir()
    for index in range(1, 6):
        record = {
            "source_candidate_id": f"candidate-{index}",
            "status": "success",
            "best_validation_f1": 0.70 + index * 0.01,
            "config": _config(f"candidate-{index}"),
        }
        (root / f"source-gate-{index}.json").write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "canonical.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.select_canonical_scratch",
            "--top5",
            str(top5_path),
            "--gate-results-root",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_candidate_id"] == "candidate-5"
    assert payload["selection_metric"] == "source_disjoint_validation_f1"


def test_stability_script_enforces_locked_gate(tmp_path: Path) -> None:
    canonical = {
        "source_candidate_id": "candidate-5",
        "config": _config("candidate-5"),
    }
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    root = tmp_path / "stability"
    root.mkdir()
    for seed, f1 in [(1, 0.90), (2, 0.89), (3, 0.885)]:
        (root / f"stability-seed-{seed}.json").write_text(
            json.dumps(
                {
                    "source_candidate_id": "candidate-5",
                    "seed": seed,
                    "status": "success",
                    "best_validation_f1": f1,
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "assessment.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.assess_scratch_stability",
            "--canonical",
            str(canonical_path),
            "--results-root",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["assessment"]["passed"] is True
