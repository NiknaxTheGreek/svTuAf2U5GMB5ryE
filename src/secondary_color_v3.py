from __future__ import annotations

import numpy as np


def _hwc_rgb(image_chw: np.ndarray) -> np.ndarray:
    a = np.asarray(image_chw)
    if a.ndim != 3 or a.shape[0] != 3 or a.dtype != np.uint8:
        raise ValueError(f"Expected uint8 CHW RGB, got {a.shape} {a.dtype}")
    return np.transpose(a, (1, 2, 0))


def _chw_rgb(image_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(image_hwc, dtype=np.uint8), (2, 0, 1))


def srgb_to_linear(rgb01: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb01, dtype=np.float64)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(linear01: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear01, dtype=np.float64), 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )


def linear_luminance(image_chw: np.ndarray) -> np.ndarray:
    rgb = _hwc_rgb(image_chw).astype(np.float64) / 255.0
    linear = srgb_to_linear(rgb)
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def linear_srgb_grayscale(image_chw: np.ndarray) -> np.ndarray:
    """
    Remove chroma while preserving CIE/Rec.709 linear-light luminance.

    The RGB triplet is converted from sRGB to linear RGB; Y is computed with
    Rec.709/sRGB primaries; the neutral RGB triplet (Y,Y,Y) is then encoded
    back to sRGB. Only final uint8 quantization prevents exact equality.
    """
    y = linear_luminance(image_chw)
    neutral_linear = np.repeat(y[..., None], 3, axis=2)
    neutral_srgb = linear_to_srgb(neutral_linear)
    out = np.clip(np.rint(neutral_srgb * 255.0), 0, 255).astype(np.uint8)
    return _chw_rgb(out)


def linear_luminance_error(
    original_chw: np.ndarray,
    transformed_chw: np.ndarray,
) -> dict[str, float]:
    a = linear_luminance(original_chw)
    b = linear_luminance(transformed_chw)
    d = np.abs(a - b)
    return {
        "mean_abs": float(np.mean(d)),
        "median_abs": float(np.median(d)),
        "p99_abs": float(np.quantile(d, 0.99)),
        "max_abs": float(np.max(d)),
    }


def chroma_residual_max(image_chw: np.ndarray) -> int:
    rgb = _hwc_rgb(image_chw).astype(np.int16)
    return int(
        max(
            np.max(np.abs(rgb[..., 0] - rgb[..., 1])),
            np.max(np.abs(rgb[..., 0] - rgb[..., 2])),
            np.max(np.abs(rgb[..., 1] - rgb[..., 2])),
        )
    )
