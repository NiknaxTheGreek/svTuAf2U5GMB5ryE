from __future__ import annotations

import numpy as np
from PIL import Image

from src.features import (
    ALL_FEATURES,
    FEATURE_SETS,
    extract_handcrafted_features,
)


def test_feature_sets_are_nested() -> None:
    assert set(FEATURE_SETS["A"]).issubset(FEATURE_SETS["B"])
    assert set(FEATURE_SETS["B"]).issubset(FEATURE_SETS["C"])
    assert tuple(FEATURE_SETS["C"]) == tuple(ALL_FEATURES)


def test_features_are_complete_finite_and_histograms_normalized() -> None:
    width, height = 64, 96
    x = np.linspace(0, 255, width, dtype=np.uint8)
    image_array = np.repeat(x[None, :], height, axis=0)
    rgb = np.stack(
        [image_array, np.flipud(image_array), image_array],
        axis=-1,
    )
    features = extract_handcrafted_features(
        Image.fromarray(rgb, mode="RGB")
    )
    assert tuple(features) == tuple(ALL_FEATURES)
    values = np.asarray(list(features.values()), dtype=float)
    assert np.isfinite(values).all()
    assert (
        abs(
            sum(
                features[f"gray_hist_{i:02d}"]
                for i in range(16)
            )
            - 1.0
        )
        < 1e-12
    )
    assert (
        abs(
            sum(
                features[f"lbp_hist_{i:02d}"]
                for i in range(16)
            )
            - 1.0
        )
        < 1e-12
    )
    assert (
        abs(
            sum(
                features[f"gradient_orientation_{i:02d}"]
                for i in range(8)
            )
            - 1.0
        )
        < 1e-12
    )


def test_uniform_image_has_zero_contrast_and_edge_density() -> None:
    image = Image.new("RGB", (32, 48), color=(128, 128, 128))
    features = extract_handcrafted_features(image)
    assert features["contrast_std"] == 0.0
    assert features["edge_density"] == 0.0
    assert features["sharpness_laplacian_var"] == 0.0
