from __future__ import annotations

import json
from pathlib import Path

from src.tuning import (
    generate_stage1_random_trials,
    scratch_parameter_count,
    stage1_payload,
    validate_stage1_payload,
)


def test_stage1_generation_is_deterministic() -> None:
    first = generate_stage1_random_trials()
    second = generate_stage1_random_trials()
    assert first == second
    assert len(first) == 40
    assert first[0].trial_id == "stage1-001"
    assert first[-1].trial_id == "stage1-040"


def test_stage1_draw_matches_frozen_distribution_facts() -> None:
    trials = generate_stage1_random_trials()
    assert sum(trial.weight_decay == 0.0 for trial in trials) == 11
    assert min(trial.parameter_count for trial in trials) == 391
    assert max(trial.parameter_count for trial in trials) == 66_455_221
    assert trials[34].trial_id == "stage1-035"
    assert trials[34].parameter_count == 66_455_221


def test_stage1_trial_ranges_and_parameter_counts() -> None:
    payload = stage1_payload()
    validate_stage1_payload(payload)
    for trial in payload["trials"]:
        assert trial["parameter_count"] == scratch_parameter_count(
            trial["depth"], trial["start_filters"]
        )


def test_committed_stage1_file_is_exact_generator_output() -> None:
    path = Path("configs/sweeps/scratch_stage1_random.json")
    committed = json.loads(path.read_text(encoding="utf-8"))
    generated = stage1_payload()
    assert committed == generated
    validate_stage1_payload(committed)
