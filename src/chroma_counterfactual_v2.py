from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

KR = 0.2126
KG = 0.7152
KB = 0.0722
CB_SCALE = 2.0 * (1.0 - KB)
CR_SCALE = 2.0 * (1.0 - KR)


@dataclass(frozen=True)
class PaletteTransferConfig:
    strength: float = 0.75
    covariance_epsilon: float = 1e-4
    stats_stride: int = 4
    gamut_iterations: int = 14


def rgb_to_ycc709(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(rgb, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = KR * r + KG * g + KB * b
    cb = (b - y) / CB_SCALE
    cr = (r - y) / CR_SCALE
    return y, cb, cr


def ycc709_to_rgb_float(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + CR_SCALE * cr
    b = y + CB_SCALE * cb
    g = (y - KR * r - KB * b) / KG
    return np.stack([r, g, b], axis=-1)


def rec709_luma(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_ycc709(rgb)[0]


def _in_gamut(rgb: np.ndarray) -> np.ndarray:
    return np.all((rgb >= 0.0) & (rgb <= 255.0), axis=-1)


def gamut_compress_chroma(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    iterations: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radially compress chroma toward neutral while preserving Y."""
    rgb = ycc709_to_rgb_float(y, cb, cr)
    direct = _in_gamut(rgb)
    lo = np.where(direct, 1.0, 0.0)
    hi = np.ones_like(y, dtype=np.float64)

    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        probe = ycc709_to_rgb_float(y, cb * mid, cr * mid)
        ok = _in_gamut(probe)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    scale = lo
    out = ycc709_to_rgb_float(y, cb * scale, cr * scale)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out, scale, direct


def rec709_grayscale(rgb: np.ndarray) -> np.ndarray:
    y = rec709_luma(rgb)
    q = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return np.repeat(q[..., None], 3, axis=-1)


def matched_chroma_rotation(
    rgb: np.ndarray,
    iterations: int = 14,
) -> tuple[np.ndarray, dict[str, float]]:
    y, cb, cr = rgb_to_ycc709(rgb)
    # +90 degrees in the (Cb, Cr) plane: (x,y) -> (-y,x)
    cb2 = -cr
    cr2 = cb
    out, scale, direct = gamut_compress_chroma(y, cb2, cr2, iterations=iterations)

    y_out = rec709_luma(out)
    metrics = {
        "luma_mae": float(np.mean(np.abs(y_out - y))),
        "gamut_compressed_fraction": float(np.mean(scale < 0.9999)),
        "mean_gamut_scale": float(np.mean(scale)),
        "direct_in_gamut_fraction": float(np.mean(direct)),
    }
    return out, metrics


def _cov_sqrt_and_invsqrt(cov: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(2, dtype=np.float64))
    vals = np.maximum(vals, eps)
    sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    invsqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return sqrt, invsqrt


def _sample_chroma(rgb: np.ndarray, stride: int) -> np.ndarray:
    _, cb, cr = rgb_to_ycc709(rgb)
    c = np.stack([cb[::stride, ::stride], cr[::stride, ::stride]], axis=-1)
    return c.reshape(-1, 2) / 128.0


def naturalistic_palette_transfer(
    target_rgb: np.ndarray,
    donor_rgb: np.ndarray,
    cfg: PaletteTransferConfig = PaletteTransferConfig(),
) -> tuple[np.ndarray, dict[str, float]]:
    y, cb, cr = rgb_to_ycc709(target_rgb)
    target_c = np.stack([cb, cr], axis=-1) / 128.0

    ts = _sample_chroma(target_rgb, cfg.stats_stride)
    ds = _sample_chroma(donor_rgb, cfg.stats_stride)

    mu_t = ts.mean(axis=0)
    mu_d = ds.mean(axis=0)
    cov_t = np.cov(ts, rowvar=False)
    cov_d = np.cov(ds, rowvar=False)

    sqrt_d, _ = _cov_sqrt_and_invsqrt(cov_d, cfg.covariance_epsilon)
    _, invsqrt_t = _cov_sqrt_and_invsqrt(cov_t, cfg.covariance_epsilon)
    transform = sqrt_d @ invsqrt_t

    flat = target_c.reshape(-1, 2)
    transferred = (flat - mu_t) @ transform.T + mu_d
    blended = (1.0 - cfg.strength) * flat + cfg.strength * transferred
    blended = blended.reshape(target_c.shape)

    cb2 = blended[..., 0] * 128.0
    cr2 = blended[..., 1] * 128.0
    out, scale, direct = gamut_compress_chroma(
        y, cb2, cr2, iterations=cfg.gamut_iterations
    )

    y_out, cb_out, cr_out = rgb_to_ycc709(out)
    displacement = np.sqrt((cb_out - cb) ** 2 + (cr_out - cr) ** 2)

    metrics = {
        "luma_mae": float(np.mean(np.abs(y_out - y))),
        "median_chroma_displacement": float(np.median(displacement)),
        "mean_chroma_displacement": float(np.mean(displacement)),
        "gamut_compressed_fraction": float(np.mean(scale < 0.9999)),
        "mean_gamut_scale": float(np.mean(scale)),
        "direct_in_gamut_fraction": float(np.mean(direct)),
    }
    return out, metrics


def stable_donor_order(target_id: str, donor_ids: list[str]) -> list[str]:
    return sorted(
        donor_ids,
        key=lambda donor_id: hashlib.sha256(
            f"{target_id}|{donor_id}".encode("utf-8")
        ).hexdigest(),
    )
