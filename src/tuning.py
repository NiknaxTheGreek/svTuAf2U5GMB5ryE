from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
STAGE1_SEED = 42
STAGE1_TRIAL_COUNT = 40
WEIGHT_DECAY_ZERO_PROBABILITY = 0.2


@dataclass(frozen=True)
class ScratchTrial:
    trial_id: str
    optimizer: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    dropout: float
    depth: int
    start_filters: int
    seed: int
    parameter_count: int


def scratch_parameter_count(depth: int, start_filters: int) -> int:
    if not 1 <= depth <= 7:
        raise ValueError("depth must be in [1, 7]")
    if not 8 <= start_filters <= 64:
        raise ValueError("start_filters must be in [8, 64]")
    total = 0
    in_channels = 3
    out_channels = start_filters
    for _ in range(depth):
        total += out_channels * in_channels * 3 * 3
        total += 2 * out_channels
        in_channels = out_channels
        out_channels *= 2
    total += in_channels + 1
    return total


def generate_stage1_random_trials(
    *,
    seed: int = STAGE1_SEED,
    trial_count: int = STAGE1_TRIAL_COUNT,
    weight_decay_zero_probability: float = WEIGHT_DECAY_ZERO_PROBABILITY,
) -> list[ScratchTrial]:
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    if not 0.0 <= weight_decay_zero_probability <= 1.0:
        raise ValueError("weight_decay_zero_probability must be in [0,1]")
    rng = np.random.default_rng(seed)
    trials: list[ScratchTrial] = []
    for index in range(1, trial_count + 1):
        optimizer = str(rng.choice(OPTIMIZERS))
        learning_rate = float(10.0 ** rng.uniform(-5.0, -2.0))
        batch_size = int(rng.integers(8, 257))
        weight_decay = (
            0.0
            if rng.random() < weight_decay_zero_probability
            else float(10.0 ** rng.uniform(-6.0, -2.0))
        )
        dropout = float(rng.uniform(0.0, 0.5))
        depth = int(rng.integers(1, 8))
        start_filters = int(rng.integers(8, 65))
        trials.append(
            ScratchTrial(
                trial_id=f"stage1-{index:03d}",
                optimizer=optimizer,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                dropout=dropout,
                depth=depth,
                start_filters=start_filters,
                seed=seed,
                parameter_count=scratch_parameter_count(depth, start_filters),
            )
        )
    return trials


def stage1_payload() -> dict[str, Any]:
    trials = generate_stage1_random_trials()
    return {
        "stage": "random",
        "seed": STAGE1_SEED,
        "trial_count": STAGE1_TRIAL_COUNT,
        "weight_decay_zero_probability": WEIGHT_DECAY_ZERO_PROBABILITY,
        "search_space": {
            "optimizer": list(OPTIMIZERS),
            "learning_rate": {
                "distribution": "log_uniform",
                "min": 1e-5,
                "max": 1e-2,
            },
            "batch_size": {
                "distribution": "integer_uniform",
                "min": 8,
                "max": 256,
            },
            "weight_decay": {
                "distribution": "mixture",
                "zero_probability": WEIGHT_DECAY_ZERO_PROBABILITY,
                "positive_distribution": "log_uniform",
                "positive_min": 1e-6,
                "positive_max": 1e-2,
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5,
            },
            "depth": {
                "distribution": "integer_uniform",
                "min": 1,
                "max": 7,
            },
            "start_filters": {
                "distribution": "integer_uniform",
                "min": 8,
                "max": 64,
            },
        },
        "fixed": {
            "seed": 42,
            "augmentation": False,
            "threshold": 0.5,
            "max_epochs": 50,
            "early_stopping_patience": 8,
            "scheduler": None,
            "gradient_clipping": None,
            "loss": "BCEWithLogitsLoss",
            "class_sampling": "natural",
            "canvas": {"width": 224, "height": 398},
            "scaling": "[0,1]",
        },
        "trials": [asdict(trial) for trial in trials],
    }


def validate_stage1_payload(payload: dict[str, Any]) -> None:
    if payload.get("stage") != "random":
        raise ValueError("Stage 1 payload must be random search")
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != STAGE1_TRIAL_COUNT:
        raise ValueError("Stage 1 must contain exactly 40 trials")
    expected_ids = [f"stage1-{index:03d}" for index in range(1, 41)]
    if [trial["trial_id"] for trial in trials] != expected_ids:
        raise ValueError("Stage 1 trial IDs are not frozen in canonical order")
    for trial in trials:
        if trial["optimizer"] not in OPTIMIZERS:
            raise ValueError("Invalid optimizer")
        if not 1e-5 <= float(trial["learning_rate"]) <= 1e-2:
            raise ValueError("Learning rate outside search space")
        if not 8 <= int(trial["batch_size"]) <= 256:
            raise ValueError("Batch size outside search space")
        weight_decay = float(trial["weight_decay"])
        if weight_decay != 0.0 and not 1e-6 <= weight_decay <= 1e-2:
            raise ValueError("Weight decay outside search space")
        if not 0.0 <= float(trial["dropout"]) <= 0.5:
            raise ValueError("Dropout outside search space")
        if not 1 <= int(trial["depth"]) <= 7:
            raise ValueError("Depth outside search space")
        if not 8 <= int(trial["start_filters"]) <= 64:
            raise ValueError("Start filters outside search space")
        expected_parameters = scratch_parameter_count(
            int(trial["depth"]), int(trial["start_filters"])
        )
        if int(trial["parameter_count"]) != expected_parameters:
            raise ValueError("Parameter count does not match architecture")



def derive_stage2_search_space(top8_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the locked Stage-2 envelope from exactly eight successful Stage-1 records."""
    if len(top8_records) != 8:
        raise ValueError("Stage 2 requires exactly eight Stage-1 records")
    if any(record.get("status") != "success" for record in top8_records):
        raise ValueError("Stage-2 envelope can use successful Stage-1 records only")

    configs = [record["config"] for record in top8_records]
    optimizers = [name for name in OPTIMIZERS if any(cfg["optimizer"] == name for cfg in configs)]
    if not optimizers:
        raise ValueError("No optimizer survived into the top eight")

    def linear_bounds(key: str, original_min: float, original_max: float) -> tuple[float, float]:
        values = [float(cfg[key]) for cfg in configs]
        expansion = 0.10 * (original_max - original_min)
        lower = max(original_min, min(values) - expansion)
        upper = min(original_max, max(values) + expansion)
        return lower, upper

    def integer_bounds(key: str, original_min: int, original_max: int) -> tuple[int, int]:
        lower, upper = linear_bounds(key, float(original_min), float(original_max))
        return max(original_min, int(np.floor(lower))), min(original_max, int(np.ceil(upper)))

    def log_bounds(
        key: str,
        original_min: float,
        original_max: float,
        *,
        positive_only: bool = False,
    ) -> tuple[float, float] | None:
        values = [float(cfg[key]) for cfg in configs]
        if positive_only:
            values = [value for value in values if value > 0.0]
            if not values:
                return None
        log_original_min = np.log10(original_min)
        log_original_max = np.log10(original_max)
        expansion = 0.10 * (log_original_max - log_original_min)
        logged = [np.log10(value) for value in values]
        lower = max(log_original_min, min(logged) - expansion)
        upper = min(log_original_max, max(logged) + expansion)
        return float(10.0**lower), float(10.0**upper)

    learning_rate = log_bounds("learning_rate", 1e-5, 1e-2)
    assert learning_rate is not None
    positive_weight_decay = log_bounds(
        "weight_decay", 1e-6, 1e-2, positive_only=True
    )
    weight_decay_zero = any(float(cfg["weight_decay"]) == 0.0 for cfg in configs)
    batch_min, batch_max = integer_bounds("batch_size", 8, 256)
    depth_min, depth_max = integer_bounds("depth", 1, 7)
    filters_min, filters_max = integer_bounds("start_filters", 8, 64)
    dropout_min, dropout_max = linear_bounds("dropout", 0.0, 0.5)

    weight_decay: dict[str, Any] = {
        "zero_retained": weight_decay_zero,
        "positive_retained": positive_weight_decay is not None,
    }
    if positive_weight_decay is not None:
        weight_decay.update(
            {
                "positive_distribution": "log_uniform",
                "positive_min": positive_weight_decay[0],
                "positive_max": positive_weight_decay[1],
            }
        )

    return {
        "optimizer": optimizers,
        "learning_rate": {
            "distribution": "log_uniform",
            "min": learning_rate[0],
            "max": learning_rate[1],
        },
        "batch_size": {
            "distribution": "integer_uniform",
            "min": batch_min,
            "max": batch_max,
        },
        "weight_decay": weight_decay,
        "dropout": {
            "distribution": "uniform",
            "min": dropout_min,
            "max": dropout_max,
        },
        "depth": {
            "distribution": "integer_uniform",
            "min": depth_min,
            "max": depth_max,
        },
        "start_filters": {
            "distribution": "integer_uniform",
            "min": filters_min,
            "max": filters_max,
        },
    }



def _sample_stage2_candidates(
    search_space: dict[str, Any],
    *,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    optimizers = list(search_space["optimizer"])
    if not optimizers:
        raise ValueError("Stage-2 optimizer set is empty")
    wd = search_space["weight_decay"]
    for _ in range(count):
        if wd["zero_retained"] and wd["positive_retained"]:
            use_zero = bool(rng.integers(0, 2))
        else:
            use_zero = bool(wd["zero_retained"])
        if use_zero:
            weight_decay = 0.0
        else:
            weight_decay = float(
                10.0
                ** rng.uniform(
                    np.log10(float(wd["positive_min"])),
                    np.log10(float(wd["positive_max"])),
                )
            )
        candidate = {
            "optimizer": str(rng.choice(optimizers)),
            "learning_rate": float(
                10.0
                ** rng.uniform(
                    np.log10(float(search_space["learning_rate"]["min"])),
                    np.log10(float(search_space["learning_rate"]["max"])),
                )
            ),
            "batch_size": int(
                rng.integers(
                    int(search_space["batch_size"]["min"]),
                    int(search_space["batch_size"]["max"]) + 1,
                )
            ),
            "weight_decay": weight_decay,
            "dropout": float(
                rng.uniform(
                    float(search_space["dropout"]["min"]),
                    float(search_space["dropout"]["max"]),
                )
            ),
            "depth": int(
                rng.integers(
                    int(search_space["depth"]["min"]),
                    int(search_space["depth"]["max"]) + 1,
                )
            ),
            "start_filters": int(
                rng.integers(
                    int(search_space["start_filters"]["min"]),
                    int(search_space["start_filters"]["max"]) + 1,
                )
            ),
        }
        candidates.append(candidate)
    return candidates


def _stage2_feature_vector(
    config: dict[str, Any],
    search_space: dict[str, Any],
) -> np.ndarray:
    optimizers = list(search_space["optimizer"])
    vector: list[float] = [
        1.0 if config["optimizer"] == optimizer else 0.0
        for optimizer in optimizers
    ]

    def scaled(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        return (value - lower) / (upper - lower)

    lr_min = np.log10(float(search_space["learning_rate"]["min"]))
    lr_max = np.log10(float(search_space["learning_rate"]["max"]))
    vector.append(
        scaled(np.log10(float(config["learning_rate"])), lr_min, lr_max)
    )
    vector.append(
        scaled(
            float(config["batch_size"]),
            float(search_space["batch_size"]["min"]),
            float(search_space["batch_size"]["max"]),
        )
    )

    wd = search_space["weight_decay"]
    is_zero = float(float(config["weight_decay"]) == 0.0)
    vector.append(is_zero)
    if wd["positive_retained"]:
        wd_min = np.log10(float(wd["positive_min"]))
        wd_max = np.log10(float(wd["positive_max"]))
        wd_value = (
            wd_min
            if is_zero
            else np.log10(float(config["weight_decay"]))
        )
        vector.append(scaled(wd_value, wd_min, wd_max))

    vector.append(
        scaled(
            float(config["dropout"]),
            float(search_space["dropout"]["min"]),
            float(search_space["dropout"]["max"]),
        )
    )
    vector.append(
        scaled(
            float(config["depth"]),
            float(search_space["depth"]["min"]),
            float(search_space["depth"]["max"]),
        )
    )
    vector.append(
        scaled(
            float(config["start_filters"]),
            float(search_space["start_filters"]["min"]),
            float(search_space["start_filters"]["max"]),
        )
    )
    return np.asarray(vector, dtype=np.float64)


def _config_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(config["optimizer"]),
        float(config["learning_rate"]),
        int(config["batch_size"]),
        float(config["weight_decay"]),
        float(config["dropout"]),
        int(config["depth"]),
        int(config["start_filters"]),
    )


def propose_stage2_bayesian_trial(
    search_space: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    iteration: int,
    seed: int = 42,
    candidate_count: int = 4096,
) -> dict[str, Any]:
    """Propose one deterministic Bayesian-optimization candidate using GP expected improvement."""
    if iteration < 1:
        raise ValueError("iteration must be positive")
    successful = [
        record
        for record in observations
        if record.get("status") == "success"
        and np.isfinite(float(record["best_validation_f1"]))
    ]
    if len(successful) < 2:
        raise ValueError("Bayesian proposal requires at least two successful observations")

    from math import erf, exp, pi, sqrt

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    x_train = np.vstack(
        [
            _stage2_feature_vector(record["config"], search_space)
            for record in successful
        ]
    )
    y_train = np.asarray(
        [float(record["best_validation_f1"]) for record in successful],
        dtype=np.float64,
    )
    seen = {_config_signature(record["config"]) for record in observations}

    candidates = _sample_stage2_candidates(
        search_space,
        seed=seed + iteration * 1009,
        count=candidate_count,
    )
    unique: list[dict[str, Any]] = []
    unique_signatures: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        signature = _config_signature(candidate)
        if signature in seen or signature in unique_signatures:
            continue
        unique_signatures.add(signature)
        unique.append(candidate)
    if not unique:
        raise RuntimeError("Bayesian candidate pool contains no unseen configurations")

    x_candidates = np.vstack(
        [_stage2_feature_vector(candidate, search_space) for candidate in unique]
    )
    dimension = x_train.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dimension),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=2,
    )
    model.fit(x_train, y_train)
    mean, std = model.predict(x_candidates, return_std=True)
    best = float(np.max(y_train))
    improvement = mean - best
    z = np.divide(
        improvement,
        std,
        out=np.zeros_like(improvement),
        where=std > 0,
    )
    cdf = np.asarray(
        [0.5 * (1.0 + erf(float(value) / sqrt(2.0))) for value in z],
        dtype=np.float64,
    )
    pdf = np.asarray(
        [exp(-0.5 * float(value) ** 2) / sqrt(2.0 * pi) for value in z],
        dtype=np.float64,
    )
    expected_improvement = improvement * cdf + std * pdf
    expected_improvement[std <= 0] = 0.0
    best_index = int(np.argmax(expected_improvement))
    proposal = dict(unique[best_index])
    proposal.update(
        {
            "trial_id": f"stage2-{iteration:03d}",
            "seed": seed,
            "parameter_count": scratch_parameter_count(
                int(proposal["depth"]),
                int(proposal["start_filters"]),
            ),
            "acquisition": "expected_improvement",
            "candidate_pool_size": candidate_count,
        }
    )
    return proposal



def select_top_combined_candidates(
    stage1_records: list[dict[str, Any]],
    stage2_records: list[dict[str, Any]],
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    successes = [
        record
        for record in [*stage1_records, *stage2_records]
        if record.get("status") == "success"
        and np.isfinite(float(record["best_validation_f1"]))
    ]
    if len(successes) < count:
        raise ValueError(
            f"Need at least {count} successful combined candidates; found {len(successes)}"
        )
    ranked = sorted(
        successes,
        key=lambda record: (
            -float(record["best_validation_f1"]),
            str(record["trial_id"]),
        ),
    )
    return ranked[:count]


def assess_three_seed_stability(
    seed_records: list[dict[str, Any]],
    *,
    std_limit: float = 0.02,
    distance_limit: float = 0.02,
) -> dict[str, Any]:
    if len(seed_records) != 3:
        raise ValueError("Stability assessment requires exactly three seed records")
    if any(record.get("status") != "success" for record in seed_records):
        raise ValueError("All three stability runs must be successful")
    seeds = [int(record["seed"]) for record in seed_records]
    if len(set(seeds)) != 3:
        raise ValueError("Stability assessment requires three distinct seeds")
    f1_values = np.asarray(
        [float(record["validation_f1"]) for record in seed_records],
        dtype=np.float64,
    )
    if not np.isfinite(f1_values).all():
        raise ValueError("Stability F1 values must be finite")
    best_f1 = float(np.max(f1_values))
    std_f1 = float(np.std(f1_values, ddof=0))
    distances = [float(best_f1 - value) for value in f1_values]
    passed = std_f1 <= std_limit and all(
        distance <= distance_limit for distance in distances
    )
    return {
        "passed": passed,
        "seed_count": 3,
        "seeds": seeds,
        "f1_values": [float(value) for value in f1_values],
        "best_f1": best_f1,
        "std_f1": std_f1,
        "max_distance_from_best": max(distances),
        "std_limit": std_limit,
        "distance_limit": distance_limit,
    }
