from __future__ import annotations

import json

import pytest
from pathlib import Path

from src.tuning import (
    derive_stage2_search_space,
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



def _success_record(**config):
    base = {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "dropout": 0.25,
        "depth": 4,
        "start_filters": 32,
    }
    base.update(config)
    return {"status": "success", "config": base}


def test_stage2_envelope_uses_exact_expansion_rules() -> None:
    records = [
        _success_record(
            optimizer="adam" if index < 4 else "sgd",
            learning_rate=10.0 ** (-4.0 + 0.1 * index),
            batch_size=50 + index,
            weight_decay=0.0 if index == 0 else 10.0 ** (-5.0 + 0.1 * index),
            dropout=0.20 + 0.01 * index,
            depth=3 + (index % 2),
            start_filters=20 + index,
        )
        for index in range(8)
    ]
    space = derive_stage2_search_space(records)
    assert space["optimizer"] == ["adam", "sgd"]
    assert space["learning_rate"]["min"] == pytest.approx(10.0 ** -4.3)
    assert space["learning_rate"]["max"] == pytest.approx(10.0 ** -3.0)
    assert space["batch_size"] == {
        "distribution": "integer_uniform",
        "min": 25,
        "max": 82,
    }
    assert space["depth"] == {
        "distribution": "integer_uniform",
        "min": 2,
        "max": 5,
    }
    assert space["start_filters"] == {
        "distribution": "integer_uniform",
        "min": 14,
        "max": 33,
    }
    assert space["dropout"]["min"] == pytest.approx(0.15)
    assert space["dropout"]["max"] == pytest.approx(0.32)
    assert space["weight_decay"]["zero_retained"] is True
    assert space["weight_decay"]["positive_retained"] is True
    assert space["weight_decay"]["positive_min"] == pytest.approx(10.0 ** -5.3)
    assert space["weight_decay"]["positive_max"] == pytest.approx(10.0 ** -3.9)


def test_stage2_drops_weight_decay_zero_when_absent() -> None:
    records = [_success_record(weight_decay=1e-4) for _ in range(8)]
    space = derive_stage2_search_space(records)
    assert space["weight_decay"]["zero_retained"] is False
    assert space["weight_decay"]["positive_retained"] is True


def test_stage2_can_retain_only_zero_weight_decay() -> None:
    records = [_success_record(weight_decay=0.0) for _ in range(8)]
    space = derive_stage2_search_space(records)
    assert space["weight_decay"] == {
        "zero_retained": True,
        "positive_retained": False,
    }
