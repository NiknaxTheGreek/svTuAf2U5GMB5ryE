from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from src.secondary_perturbations_v2 import (
    HandMaskConfig,
    skin_hand_mask,
)


@dataclass(frozen=True)
class MediaPipeMaskConfig:
    name: str
    hull_dilate_kernel: int
    skin_neighborhood_kernel: int = 0
    use_skin_extension: bool = False
    fallback_classical: bool = False


MEDIAPIPE_MASK_CANDIDATES: tuple[MediaPipeMaskConfig, ...] = (
    MediaPipeMaskConfig(
        name="mp_hull_d09",
        hull_dilate_kernel=9,
    ),
    MediaPipeMaskConfig(
        name="mp_hull_d15",
        hull_dilate_kernel=15,
    ),
    MediaPipeMaskConfig(
        name="mp_seed_skin",
        hull_dilate_kernel=9,
        skin_neighborhood_kernel=51,
        use_skin_extension=True,
    ),
    MediaPipeMaskConfig(
        name="mp_seed_skin_fallback",
        hull_dilate_kernel=9,
        skin_neighborhood_kernel=51,
        use_skin_extension=True,
        fallback_classical=True,
    ),
)


_STRICT_SPATIAL = HandMaskConfig(
    name="strict_spatial_for_mp_seed",
    cb_min=70,
    cb_max=130,
    cr_min=136,
    cr_max=178,
    y_min=30,
    rgb_gate="strict",
    boundary_mode="none",
    min_component_pixels=35,
    max_component_fraction=0.30,
    open_kernel=3,
    close_kernel=5,
    dilate_kernel=3,
    min_y_fraction=0.08,
)

_STRICT_LOWER_SIDE = HandMaskConfig(
    name="strict_lower_side_fallback",
    cb_min=70,
    cb_max=130,
    cr_min=136,
    cr_max=178,
    y_min=30,
    rgb_gate="strict",
    boundary_mode="lower_sides",
    min_component_pixels=50,
    max_component_fraction=0.30,
    open_kernel=3,
    close_kernel=5,
    dilate_kernel=5,
    min_y_fraction=0.15,
)


def _hwc_rgb(image_chw: np.ndarray) -> np.ndarray:
    array = np.asarray(image_chw)
    if array.ndim != 3 or array.shape[0] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 CHW RGB image, got {array.shape} {array.dtype}")
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def _cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _dilate(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or not np.any(mask):
        return mask.copy()
    if kernel % 2 == 0:
        raise ValueError("Dilation kernel must be odd")
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(kernel)), dtype=np.uint8) >= 128


def _landmark_hull_mask(
    hand_landmarks,
    height: int,
    width: int,
) -> np.ndarray:
    points: list[tuple[int, int]] = []
    for lm in hand_landmarks:
        x = int(round(float(lm.x) * (width - 1)))
        y = int(round(float(lm.y) * (height - 1)))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        points.append((x, y))
    hull = convex_hull(points)
    mask_img = Image.new("L", (width, height), 0)
    if len(hull) >= 3:
        ImageDraw.Draw(mask_img).polygon(hull, fill=255)
    elif len(hull) == 2:
        ImageDraw.Draw(mask_img).line(hull, fill=255, width=3)
    elif len(hull) == 1:
        ImageDraw.Draw(mask_img).point(hull[0], fill=255)
    return np.asarray(mask_img, dtype=np.uint8) >= 128


class MediaPipeHandMasker:
    def __init__(
        self,
        model_path: str | Path,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        num_hands: int = 2,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self.model_path = str(Path(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeHandMasker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _detect(self, rgb: np.ndarray):
        array = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=array,
        )
        return self._landmarker.detect(mp_image)

    def detect_seed(self, image_chw: np.ndarray) -> tuple[np.ndarray, int]:
        rgb = _hwc_rgb(image_chw)
        result = self._detect(rgb)
        height, width = rgb.shape[:2]
        seed = np.zeros((height, width), dtype=bool)
        for hand in result.hand_landmarks:
            seed |= _landmark_hull_mask(hand, height, width)
        return seed, len(result.hand_landmarks)

    def detect_seed_from_source_rgb(
        self,
        source_rgb: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> tuple[np.ndarray, int]:
        """Detect on the original RGB frame, map normalized landmarks to model canvas."""
        rgb = np.ascontiguousarray(source_rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB source image, got {rgb.shape}")

        result = self._detect(rgb)
        source_height, source_width = rgb.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2

        seed = np.zeros((target_height, target_width), dtype=bool)
        for hand in result.hand_landmarks:
            points: list[tuple[int, int]] = []
            for lm in hand:
                x = pad_left + int(round(float(lm.x) * (resized_width - 1)))
                y = pad_top + int(round(float(lm.y) * (resized_height - 1)))
                x = max(0, min(target_width - 1, x))
                y = max(0, min(target_height - 1, y))
                points.append((x, y))
            hull = convex_hull(points)
            mask_img = Image.new("L", (target_width, target_height), 0)
            if len(hull) >= 3:
                ImageDraw.Draw(mask_img).polygon(hull, fill=255)
            elif len(hull) == 2:
                ImageDraw.Draw(mask_img).line(hull, fill=255, width=3)
            elif len(hull) == 1:
                ImageDraw.Draw(mask_img).point(hull[0], fill=255)
            seed |= np.asarray(mask_img, dtype=np.uint8) >= 128
        return seed, len(result.hand_landmarks)

    def _candidate_masks_from_seed(
        self,
        image_chw: np.ndarray,
        seed: np.ndarray,
    ) -> dict[str, np.ndarray]:
        strict_skin = skin_hand_mask(image_chw, _STRICT_SPATIAL)
        fallback = skin_hand_mask(image_chw, _STRICT_LOWER_SIDE)

        masks: dict[str, np.ndarray] = {}
        for cfg in MEDIAPIPE_MASK_CANDIDATES:
            hull = _dilate(seed, cfg.hull_dilate_kernel)
            mask = hull.copy()
            if cfg.use_skin_extension and np.any(seed):
                neighborhood = _dilate(seed, cfg.skin_neighborhood_kernel)
                mask |= strict_skin & neighborhood
                mask = _dilate(mask, 5)
            if cfg.fallback_classical and not np.any(seed):
                mask |= fallback
            masks[cfg.name] = mask
        return masks

    def candidate_masks(self, image_chw: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
        seed, detected_hands = self.detect_seed(image_chw)
        return self._candidate_masks_from_seed(image_chw, seed), detected_hands

    def candidate_masks_from_source_rgb(
        self,
        image_chw: np.ndarray,
        source_rgb: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], int]:
        _, target_height, target_width = image_chw.shape
        seed, detected_hands = self.detect_seed_from_source_rgb(
            source_rgb,
            target_height=target_height,
            target_width=target_width,
        )
        return self._candidate_masks_from_seed(image_chw, seed), detected_hands
