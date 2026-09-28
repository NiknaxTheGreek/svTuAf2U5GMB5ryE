from __future__ import annotations

import math
from io import BytesIO
from typing import Mapping, Sequence
from zipfile import ZipFile

import numpy as np
from PIL import Image


FEATURE_SIZE = (224, 398)
EDGE_THRESHOLD = 0.08

SET_A = (
    "brightness_mean",
    "contrast_std",
    "sharpness_laplacian_var",
    "entropy_bits",
    "edge_density",
)

SET_B_EXTRA = (
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "saturation_mean", "saturation_std",
    "hue_sin_mean", "hue_cos_mean",
    "gray_p10", "gray_p25", "gray_median", "gray_p75", "gray_p90",
)

GRAY_HIST = tuple(f"gray_hist_{index:02d}" for index in range(16))
LBP_HIST = tuple(f"lbp_hist_{index:02d}" for index in range(16))
ORIENTATION_HIST = tuple(f"gradient_orientation_{index:02d}" for index in range(8))
SET_C_EXTRA = GRAY_HIST + LBP_HIST + ORIENTATION_HIST + (
    "gradient_magnitude_mean",
    "gradient_magnitude_std",
)

FEATURE_SETS = {
    "A": SET_A,
    "B": SET_A + SET_B_EXTRA,
    "C": SET_A + SET_B_EXTRA + SET_C_EXTRA,
}
ALL_FEATURES = FEATURE_SETS["C"]


def _normalized_histogram(
    values: np.ndarray, *, bins: int, value_range: tuple[float, float]
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot normalize an empty histogram")
    return counts.astype(np.float64) / float(total)


def _lbp_codes(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.ndim != 2 or min(gray_u8.shape) < 3:
        raise ValueError("LBP requires a grayscale image at least 3x3")
    center = gray_u8[1:-1, 1:-1]
    neighbors = (
        gray_u8[:-2, :-2],
        gray_u8[:-2, 1:-1],
        gray_u8[:-2, 2:],
        gray_u8[1:-1, 2:],
        gray_u8[2:, 2:],
        gray_u8[2:, 1:-1],
        gray_u8[2:, :-2],
        gray_u8[1:-1, :-2],
    )
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    return codes


def extract_handcrafted_features(image: Image.Image) -> dict[str, float]:
    rgb_image = image.convert("RGB").resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float64) / 255.0
    gray_image = rgb_image.convert("L")
    gray_u8 = np.asarray(gray_image, dtype=np.uint8)
    gray = gray_u8.astype(np.float64) / 255.0

    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    gray_hist_256 = _normalized_histogram(gray, bins=256, value_range=(0.0, 1.0))
    nonzero = gray_hist_256[gray_hist_256 > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.float64) / 255.0
    hue_angle = hsv[..., 0] * (2.0 * math.pi)

    features: dict[str, float] = {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness_laplacian_var": float(laplacian.var()),
        "entropy_bits": entropy,
        "edge_density": float(np.mean(magnitude > EDGE_THRESHOLD)),
        "rgb_mean_r": float(rgb[..., 0].mean()),
        "rgb_mean_g": float(rgb[..., 1].mean()),
        "rgb_mean_b": float(rgb[..., 2].mean()),
        "rgb_std_r": float(rgb[..., 0].std()),
        "rgb_std_g": float(rgb[..., 1].std()),
        "rgb_std_b": float(rgb[..., 2].std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "saturation_std": float(hsv[..., 1].std()),
        "hue_sin_mean": float(np.sin(hue_angle).mean()),
        "hue_cos_mean": float(np.cos(hue_angle).mean()),
        "gray_p10": float(np.quantile(gray, 0.10)),
        "gray_p25": float(np.quantile(gray, 0.25)),
        "gray_median": float(np.quantile(gray, 0.50)),
        "gray_p75": float(np.quantile(gray, 0.75)),
        "gray_p90": float(np.quantile(gray, 0.90)),
        "gradient_magnitude_mean": float(magnitude.mean()),
        "gradient_magnitude_std": float(magnitude.std()),
    }

    gray_hist = _normalized_histogram(gray, bins=16, value_range=(0.0, 1.0))
    for name, value in zip(GRAY_HIST, gray_hist, strict=True):
        features[name] = float(value)

    lbp = _lbp_codes(gray_u8)
    lbp_hist = _normalized_histogram(lbp, bins=16, value_range=(0.0, 256.0))
    for name, value in zip(LBP_HIST, lbp_hist, strict=True):
        features[name] = float(value)

    orientation_hist, _ = np.histogram(
        orientation,
        bins=8,
        range=(-math.pi, math.pi),
        weights=magnitude,
    )
    orientation_total = float(orientation_hist.sum())
    if orientation_total > 0:
        orientation_hist = orientation_hist.astype(np.float64) / orientation_total
    else:
        orientation_hist = np.full(8, 1.0 / 8.0, dtype=np.float64)
    for name, value in zip(ORIENTATION_HIST, orientation_hist, strict=True):
        features[name] = float(value)

    missing = set(ALL_FEATURES) - features.keys()
    if missing:
        raise RuntimeError(f"Feature extraction omitted columns: {sorted(missing)}")
    values = np.asarray([features[name] for name in ALL_FEATURES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature extraction produced NaN or infinite values")
    return {name: features[name] for name in ALL_FEATURES}


def extract_feature_rows(
    archive_path: str,
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    verify_pixel_hashes: bool = True,
) -> list[dict[str, object]]:
    from src.data import decoded_pixel_sha256

    rows: list[dict[str, object]] = []
    with ZipFile(archive_path) as archive:
        for manifest_row in manifest_rows:
            encoded = archive.read(manifest_row["archive_member"])
            with Image.open(BytesIO(encoded)) as image:
                rgb = image.convert("RGB")
                if verify_pixel_hashes:
                    observed = decoded_pixel_sha256(rgb)
                    expected = manifest_row["pixel_sha256"]
                    if observed != expected:
                        raise ValueError(f"Pixel hash mismatch for {manifest_row['sample_id']}")
                features = extract_handcrafted_features(rgb)
            row: dict[str, object] = {
                "sample_id": manifest_row["sample_id"],
                "label": manifest_row["label"],
                "video_id": manifest_row["video_id"],
                "frame_number": int(manifest_row["frame_number"]),
                "supplied_split": manifest_row["supplied_split"],
                "pixel_sha256": manifest_row["pixel_sha256"],
            }
            row.update(features)
            rows.append(row)
    return rows
