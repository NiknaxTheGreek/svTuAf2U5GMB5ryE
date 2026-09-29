from __future__ import annotations

import json

import pytest
from pathlib import Path

from src.tuning import (
    derive_stage2_search_space,
    generate_stage1_random_trials,
    propose_stage2_bayesian_trial,
    select_top_combined_candidates,
    assess_three_seed_stability,
    scratch_parameter_count,
    stage1_payload,
    validate_stage1_payload,
)


def test_stage1_generation_is_deterministic() -> None:
    first = generate_stage1_random_trials()
    second = generate_stage1_random_trials()
    assert first == second
    assert len(first) == 12
    assert first[0].trial_id == "stage1-001"
    assert first[-1].trial_id == "stage1-012"


def test_stage1_draw_matches_frozen_distribution_facts() -> None:
    trials = generate_stage1_random_trials()
    assert sum(trial.weight_decay == 0.0 for trial in trials) == 4
    assert min(trial.parameter_count for trial in trials) == 7_696
    assert max(trial.parameter_count for trial in trials) == 1_202_461
    assert trials[7].trial_id == "stage1-008"
    assert trials[7].parameter_count == 1_202_461


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
            optimizer="adam" if index < 4 else "rmsprop",
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
    assert space["optimizer"] == ["adam", "rmsprop"]
    assert space["learning_rate"]["min"] == pytest.approx(1e-4)
    assert space["learning_rate"]["max"] == pytest.approx(10.0 ** -3.1522878745280337)
    assert space["batch_size"] == {
        "distribution": "integer_uniform",
        "min": 40,
        "max": 67,
    }
    assert space["depth"] == {
        "distribution": "integer_uniform",
        "min": 2,
        "max": 5,
    }
    assert space["start_filters"] == {
        "distribution": "integer_uniform",
        "min": 17,
        "max": 30,
    }
    assert space["dropout"]["min"] == pytest.approx(0.165)
    assert space["dropout"]["max"] == pytest.approx(0.305)
    assert space["weight_decay"]["zero_retained"] is True
    assert space["weight_decay"]["positive_retained"] is True
    assert space["weight_decay"]["positive_min"] == pytest.approx(10.0 ** -5.2)
    assert space["weight_decay"]["positive_max"] == pytest.approx(1e-4)


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



def test_stage2_bayesian_proposal_is_deterministic_and_unseen() -> None:
    records = [
        _success_record(
            optimizer="adam" if index < 4 else "rmsprop",
            learning_rate=10.0 ** (-4.0 + 0.1 * index),
            batch_size=50 + index,
            weight_decay=0.0 if index == 0 else 10.0 ** (-5.0 + 0.1 * index),
            dropout=0.20 + 0.01 * index,
            depth=3 + (index % 2),
            start_filters=20 + index,
        )
        | {
            "trial_id": f"stage1-{index + 1:03d}",
            "best_validation_f1": 0.70 + index * 0.01,
        }
        for index in range(8)
    ]
    space = derive_stage2_search_space(records)
    first = propose_stage2_bayesian_trial(
        space, records, iteration=1, candidate_count=256
    )
    second = propose_stage2_bayesian_trial(
        space, records, iteration=1, candidate_count=256
    )
    assert first == second
    assert first["trial_id"] == "stage2-001"
    assert first["optimizer"] in space["optimizer"]
    assert not any(
        (
            first["optimizer"],
            first["learning_rate"],
            first["batch_size"],
            first["weight_decay"],
            first["dropout"],
            first["depth"],
            first["start_filters"],
        )
        == (
            record["config"]["optimizer"],
            record["config"]["learning_rate"],
            record["config"]["batch_size"],
            record["config"]["weight_decay"],
            record["config"]["dropout"],
            record["config"]["depth"],
            record["config"]["start_filters"],
        )
        for record in records
    )
    assert first["parameter_count"] == scratch_parameter_count(
        first["depth"], first["start_filters"]
    )



def test_combined_top_five_uses_validation_f1_then_trial_id() -> None:
    stage1 = [
        {
            "trial_id": "stage1-001",
            "status": "success",
            "best_validation_f1": 0.80,
        },
        {
            "trial_id": "stage1-002",
            "status": "failed",
            "best_validation_f1": 0.99,
        },
        {
            "trial_id": "stage1-003",
            "status": "success",
            "best_validation_f1": 0.85,
        },
    ]
    stage2 = [
        {
            "trial_id": f"stage2-{index:03d}",
            "status": "success",
            "best_validation_f1": value,
        }
        for index, value in enumerate(
            [0.90, 0.88, 0.85, 0.83, 0.81], start=1
        )
    ]
    top = select_top_combined_candidates(stage1, stage2, count=5)
    assert [record["trial_id"] for record in top] == [
        "stage2-001",
        "stage2-002",
        "stage1-003",
        "stage2-003",
        "stage2-004",
    ]


def test_three_seed_stability_gate_passes_locked_rule() -> None:
    records = [
        {"seed": 1, "status": "success", "validation_f1": 0.90},
        {"seed": 2, "status": "success", "validation_f1": 0.89},
        {"seed": 3, "status": "success", "validation_f1": 0.885},
    ]
    result = assess_three_seed_stability(records)
    assert result["passed"] is True
    assert result["std_f1"] <= 0.02
    assert result["max_distance_from_best"] <= 0.02


def test_three_seed_stability_gate_fails_distance_rule() -> None:
    records = [
        {"seed": 1, "status": "success", "validation_f1": 0.90},
        {"seed": 2, "status": "success", "validation_f1": 0.89},
        {"seed": 3, "status": "success", "validation_f1": 0.875},
    ]
    result = assess_three_seed_stability(records)
    assert result["passed"] is False
    assert result["max_distance_from_best"] > 0.02
