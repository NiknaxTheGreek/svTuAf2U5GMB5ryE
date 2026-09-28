from __future__ import annotations

import json
import subprocess
import sys


def test_stage2_generator_requires_complete_stage1_top8(tmp_path) -> None:
    results = []
    for index in range(1, 41):
        results.append(
            {
                "trial_id": f"stage1-{index:03d}",
                "status": "success",
                "best_validation_f1": 1.0 - index / 1000.0,
                "config": {
                    "optimizer": "adam" if index <= 4 else "sgd",
                    "learning_rate": 1e-4,
                    "batch_size": 64,
                    "weight_decay": 1e-4,
                    "dropout": 0.2,
                    "depth": 3,
                    "start_filters": 24,
                },
            }
        )
    payload = {
        "stage": "random",
        "trial_count": 40,
        "top8_trial_ids": [f"stage1-{index:03d}" for index in range(1, 9)],
        "results": results,
    }
    source = tmp_path / "stage1.json"
    output = tmp_path / "stage2.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.derive_scratch_stage2",
            "--stage1-results",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["stage"] == "bayesian"
    assert generated["trial_count"] == 30
    assert len(generated["source_stage1_top8"]) == 8
    assert generated["search_space"]["optimizer"] == ["adam", "sgd"]
