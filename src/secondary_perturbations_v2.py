from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class HandMaskConfig:
    name: str
    cb_min: int
    cb_max: int
    cr_min: int
    cr_max: int
    y_min: int
    require_rgb_gate: bool
    boundary_only: bool
    min_component_pixels: int = 80
    open_kernel: int = 3
    close_kernel: int = 5
    dilate_kernel: int = 7


HAND_MASK_CANDIDATES: tuple[HandMaskConfig, ...] = (
    HandMaskConfig(
        name="ycbcr_core",
        cb_min=77, cb_max=127, cr_min=133, cr_max=173, y_min=40,
        require_rgb_gate=False, boundary_only=False,
    ),
    HandMaskConfig(
        name="ycbcr_wide_rgb_gate",
        cb_min=65, cb_max=135, cr_min=125, cr_max=180, y_min=30,
        require_rgb_gate=True, boundary_only=False,
    ),
    HandMaskConfig(
        name="ycbcr_wide_rgb_boundary",
        cb_min=65, cb_max=135, cr_min=125, cr_max=180, y_min=30,
        require_rgb_gate=True, boundary_only=True,
    ),
    HandMaskConfig(
        name="consensus_boundary",
        cb_min=70, cb_max=132, cr_min=128, cr_max=178, y_min=32,
        require_rgb_gate=False, boundary_only=True,
    ),
)


def _hwc_rgb(image_chw: np.ndarray) -> np.ndarray:
    array = np.asarray(image_chw)
    if array.shape[0] != 3 or array.ndim != 3:
        raise ValueError(f"Expected CHW RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {array.dtype}")
    return np.transpose(array, (1, 2, 0))


def _chw_rgb(image_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(image_hwc, dtype=np.uint8), (2, 0, 1))


def rec709_grayscale(image_chw: np.ndarray) -> np.ndarray:
    """Remove chroma while preserving gamma-encoded Rec.709 luma by definition."""
    rgb = _hwc_rgb(image_chw).astype(np.float32)
    y = np.rint(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    y = np.clip(y, 0, 255).astype(np.uint8)
    out = np.repeat(y[..., None], 3, axis=2)
    return _chw_rgb(out)


def pil_l_grayscale(image_chw: np.ndarray) -> np.ndarray:
    """Comparator only: Pillow's standard L conversion, replicated to three channels."""
    rgb = _hwc_rgb(image_chw)
    gray = np.asarray(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)
    return _chw_rgb(np.repeat(gray[..., None], 3, axis=2))


def rec709_luma(image_chw: np.ndarray) -> np.ndarray:
    rgb = _hwc_rgb(image_chw).astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def grayscale_luma_mae(original_chw: np.ndarray, transformed_chw: np.ndarray) -> float:
    return float(np.mean(np.abs(rec709_luma(original_chw) - rec709_luma(transformed_chw))))


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _normalized_rgb_skin_gate(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=2)
    mn = np.min(x, axis=2)
    daylight = (
        (r > 95) & (g > 40) & (b > 20) &
        ((mx - mn) > 15) & (np.abs(r - g) > 15) &
        (r > g) & (r > b)
    )
    bright = (
        (r > 200) & (g > 180) & (b > 160) &
        (np.abs(r - g) <= 30) & (r > b) & (g > b)
    )
    return daylight | bright


def _morph(mask: np.ndarray, config: HandMaskConfig) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if config.open_kernel > 1:
        image = image.filter(ImageFilter.MinFilter(config.open_kernel))
        image = image.filter(ImageFilter.MaxFilter(config.open_kernel))
    if config.close_kernel > 1:
        image = image.filter(ImageFilter.MaxFilter(config.close_kernel))
        image = image.filter(ImageFilter.MinFilter(config.close_kernel))
    return np.asarray(image, dtype=np.uint8) >= 128


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            queue = deque([(y, x)])
            seen[y, x] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            components.append(np.asarray(coords, dtype=np.int32))
    return components


def _component_touches_boundary(coords: np.ndarray, h: int, w: int, margin: int = 4) -> bool:
    ys, xs = coords[:, 0], coords[:, 1]
    return bool(
        np.any(ys <= margin) or np.any(ys >= h - 1 - margin) or
        np.any(xs <= margin) or np.any(xs >= w - 1 - margin)
    )


def skin_hand_mask(image_chw: np.ndarray, config: HandMaskConfig) -> np.ndarray:
    rgb = _hwc_rgb(image_chw)
    y, cb, cr = _rgb_to_ycbcr(rgb)
    mask = (
        (y >= config.y_min) &
        (cb >= config.cb_min) & (cb <= config.cb_max) &
        (cr >= config.cr_min) & (cr <= config.cr_max)
    )
    if config.require_rgb_gate:
        mask &= _normalized_rgb_skin_gate(rgb)
    mask = _morph(mask, config)

    cleaned = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for coords in _connected_components(mask):
        if len(coords) < config.min_component_pixels:
            continue
        if config.boundary_only and not _component_touches_boundary(coords, h, w):
            continue
        cleaned[coords[:, 0], coords[:, 1]] = True

    if config.dilate_kernel > 1 and np.any(cleaned):
        image = Image.fromarray((cleaned.astype(np.uint8) * 255), mode="L")
        image = image.filter(ImageFilter.MaxFilter(config.dilate_kernel))
        cleaned = np.asarray(image, dtype=np.uint8) >= 128
    return cleaned


def apply_hand_mask(
    image_chw: np.ndarray,
    mask: np.ndarray,
    fill: str = "local_border_median",
) -> np.ndarray:
    rgb = _hwc_rgb(image_chw).copy()
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask/image shape mismatch")
    if not np.any(mask):
        return image_chw.copy()

    if fill == "black":
        rgb[mask] = 0
    elif fill == "global_median":
        background = rgb[~mask]
        value = np.median(background, axis=0) if len(background) else np.array([127, 127, 127])
        rgb[mask] = np.rint(value).astype(np.uint8)
    elif fill == "local_border_median":
        m = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        border = np.asarray(m.filter(ImageFilter.MaxFilter(15)), dtype=np.uint8) >= 128
        ring = border & ~mask
        pixels = rgb[ring]
        if len(pixels) < 50:
            pixels = rgb[~mask]
        value = np.median(pixels, axis=0) if len(pixels) else np.array([127, 127, 127])
        rgb[mask] = np.rint(value).astype(np.uint8)
    else:
        raise ValueError(f"Unknown fill mode: {fill}")
    return _chw_rgb(rgb)


def apply_matched_control_mask(image_chw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate the same mask to the opposite horizontal side; preserve area exactly when possible."""
    mirrored = np.fliplr(mask)
    transformed = apply_hand_mask(image_chw, mirrored, fill="local_border_median")
    return transformed, mirrored


def overlay_mask(image_chw: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = _hwc_rgb(image_chw).astype(np.float32)
    overlay = rgb.copy()
    overlay[mask] = np.array([255.0, 0.0, 0.0])
    out = rgb * (1.0 - alpha) + overlay * alpha
    return _chw_rgb(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def mask_summary(mask: np.ndarray) -> dict[str, float | int | bool]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    coverage = float(mask.mean())
    if len(xs) == 0:
        return {
            "pixels": 0,
            "coverage_fraction": 0.0,
            "touches_boundary": False,
            "bbox_x0": -1, "bbox_y0": -1, "bbox_x1": -1, "bbox_y1": -1,
        }
    return {
        "pixels": int(len(xs)),
        "coverage_fraction": coverage,
        "touches_boundary": bool(np.any(ys == 0) or np.any(ys == h - 1) or np.any(xs == 0) or np.any(xs == w - 1)),
        "bbox_x0": int(xs.min()), "bbox_y0": int(ys.min()),
        "bbox_x1": int(xs.max()), "bbox_y1": int(ys.max()),
    }
