from __future__ import annotations

import numpy as np

from src.training import (
    fit_logistic_classifier,
    positive_probabilities,
    standardized_logistic_coefficients,
)


def test_logistic_pipeline_scales_and_separates_simple_data() -> None:
    x = np.array(
        [[-2.0, 0.0], [-1.0, 0.1], [1.0, 0.0], [2.0, -0.1]]
    )
    y = np.array([0, 0, 1, 1])
    model = fit_logistic_classifier(x, y, c_value=1.0)
    probabilities = positive_probabilities(model, x)
    assert probabilities.shape == (4,)
    assert probabilities[:2].max() < 0.5
    assert probabilities[2:].min() >= 0.5
    coefficients = standardized_logistic_coefficients(
        model, ["x1", "x2"]
    )
    assert [name for name, _ in coefficients] == ["x1", "x2"]
