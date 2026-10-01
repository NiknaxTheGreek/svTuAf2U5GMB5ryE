from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class ViewSpec:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    flip_x: bool = False


@dataclass(frozen=True)
class Detection:
    view: str
    score: float
    points_target: np.ndarray  # (21,2), x/y on model canvas


@dataclass(frozen=True)
class GeometryConfig:
    name: str
    confidence: float
    hull_dilate: int
    final_dilate: int
    include_forearm: bool
    forearm_width_scale_near: float = 0.48
    forearm_width_scale_far: float = 0.62
    max_forearm_length_fraction: float = 0.52


GEOMETRY_CANDIDATES: tuple[GeometryConfig, ...] = (
    GeometryConfig(
        name="mv50_hull",
        confidence=0.50,
        hull_dilate=11,
        final_dilate=3,
        include_forearm=False,
    ),
    GeometryConfig(
        name="mv50_handarm",
        confidence=0.50,
        hull_dilate=11,
        final_dilate=3,
        include_forearm=True,
    ),
    GeometryConfig(
        name="mv35_handarm",
        confidence=0.35,
        hull_dilate=11,
        final_dilate=3,
        include_forearm=True,
    ),
)


def _convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    pts = sorted(set((int(round(x)), int(round(y))) for x, y in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _dilate(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or not np.any(mask):
        return mask.copy()
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd")
    im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(im.filter(ImageFilter.MaxFilter(kernel)), dtype=np.uint8) >= 128


def _views() -> tuple[ViewSpec, ...]:
    return (
        ViewSpec("full", 0.00, 0.00, 1.00, 1.00, False),
        ViewSpec("full_mirror", 0.00, 0.00, 1.00, 1.00, True),
        ViewSpec("left72", 0.00, 0.00, 0.72, 1.00, False),
        ViewSpec("right72", 0.28, 0.00, 1.00, 1.00, False),
        ViewSpec("bottom78", 0.00, 0.22, 1.00, 1.00, False),
        ViewSpec("lower_left", 0.00, 0.20, 0.72, 1.00, False),
        ViewSpec("lower_right", 0.28, 0.20, 1.00, 1.00, False),
    )


def _source_crop(rgb: np.ndarray, spec: ViewSpec) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    h, w = rgb.shape[:2]
    x0 = int(round(spec.x0 * (w - 1)))
    x1 = int(round(spec.x1 * (w - 1))) + 1
    y0 = int(round(spec.y0 * (h - 1)))
    y1 = int(round(spec.y1 * (h - 1))) + 1
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    if spec.flip_x:
        crop = np.ascontiguousarray(crop[:, ::-1])
    return crop, (x0, y0, x1, y1)


def _source_to_target(
    xy_source: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scale = min(target_width / source_width, target_height / source_height)
    resized_w = max(1, min(target_width, round(source_width * scale)))
    resized_h = max(1, min(target_height, round(source_height * scale)))
    pad_left = (target_width - resized_w) // 2
    pad_top = (target_height - resized_h) // 2

    out = np.empty_like(xy_source, dtype=np.float64)
    out[:, 0] = pad_left + xy_source[:, 0] * (resized_w - 1) / max(source_width - 1, 1)
    out[:, 1] = pad_top + xy_source[:, 1] * (resized_h - 1) / max(source_height - 1, 1)
    return out


def _map_landmarks_to_source(hand, spec: ViewSpec, crop_box: tuple[int,int,int,int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    cw = x1 - x0
    ch = y1 - y0
    pts = []
    for lm in hand:
        xv = float(lm.x) * max(cw - 1, 1)
        yv = float(lm.y) * max(ch - 1, 1)
        if spec.flip_x:
            xv = (cw - 1) - xv
        pts.append((x0 + xv, y0 + yv))
    return np.asarray(pts, dtype=np.float64)


def _palm_width(points: np.ndarray) -> float:
    # Index MCP to pinky MCP spans palm width robustly.
    return float(np.linalg.norm(points[5] - points[17]))


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points[[0,5,9,13,17]], axis=0)


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    """
    Collapse the same hand detected in several crops/views.
    Prefer higher-confidence detections; full/full_mirror win close ties.
    """
    priority = {"full": 3, "full_mirror": 2}
    ordered = sorted(
        detections,
        key=lambda d: (d.score, priority.get(d.view, 1)),
        reverse=True,
    )
    kept: list[Detection] = []
    for det in ordered:
        c = _centroid(det.points_target)
        w = max(_palm_width(det.points_target), 8.0)
        duplicate = False
        for prev in kept:
            cp = _centroid(prev.points_target)
            wp = max(_palm_width(prev.points_target), 8.0)
            if np.linalg.norm(c - cp) <= 0.75 * max(w, wp):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _polygon_mask(points: list[tuple[float,float]], h: int, w: int) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    poly = _convex_hull(points)
    if len(poly) >= 3:
        ImageDraw.Draw(im).polygon(poly, fill=255)
    elif len(poly) == 2:
        ImageDraw.Draw(im).line(poly, fill=255, width=3)
    elif len(poly) == 1:
        ImageDraw.Draw(im).point(poly[0], fill=255)
    return np.asarray(im, dtype=np.uint8) >= 128


def _forearm_polygon(points: np.ndarray, h: int, w: int, cfg: GeometryConfig) -> list[tuple[float,float]]:
    wrist = points[0].astype(np.float64)
    palm_center = np.mean(points[[5,9,13,17]], axis=0)
    direction = wrist - palm_center
    n = float(np.linalg.norm(direction))
    if n < 1e-6:
        return []
    direction /= n

    # Ray from wrist out of the palm to the first image boundary.
    candidates = []
    dx, dy = float(direction[0]), float(direction[1])
    if dx > 1e-9:
        candidates.append(((w - 1 - wrist[0]) / dx))
    elif dx < -1e-9:
        candidates.append((0 - wrist[0]) / dx)
    if dy > 1e-9:
        candidates.append(((h - 1 - wrist[1]) / dy))
    elif dy < -1e-9:
        candidates.append((0 - wrist[1]) / dy)
    candidates = [t for t in candidates if t > 0]
    if not candidates:
        return []
    t = min(candidates)
    length = float(t)
    max_len = cfg.max_forearm_length_fraction * min(h, w)
    if length > max_len:
        # If the predicted arm direction does not exit near the wrist, avoid
        # drawing a long corridor through the page.
        return []

    palm_w = max(_palm_width(points), 8.0)
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    near_half = cfg.forearm_width_scale_near * palm_w
    far_half = cfg.forearm_width_scale_far * palm_w
    far = wrist + direction * length

    poly = [
        wrist + perp * near_half,
        wrist - perp * near_half,
        far - perp * far_half,
        far + perp * far_half,
    ]
    clipped = []
    for p in poly:
        clipped.append((
            float(np.clip(p[0], 0, w - 1)),
            float(np.clip(p[1], 0, h - 1)),
        ))
    return clipped


def geometry_mask(detections: list[Detection], h: int, w: int, cfg: GeometryConfig) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        hand = _polygon_mask([tuple(p) for p in det.points_target], h, w)
        hand = _dilate(hand, cfg.hull_dilate)
        mask |= hand
        if cfg.include_forearm:
            arm_poly = _forearm_polygon(det.points_target, h, w, cfg)
            if arm_poly:
                mask |= _polygon_mask(arm_poly, h, w)
    return _dilate(mask, cfg.final_dilate)


class MultiViewMediaPipeHandMasker:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float,
        num_hands: int = 2,
    ) -> None:
        import mediapipe as mp
        self._mp = mp
        self.confidence = float(confidence)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model_path))),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=self.confidence,
            min_hand_presence_confidence=self.confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _detect(self, rgb: np.ndarray):
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb, dtype=np.uint8),
        )
        return self._landmarker.detect(image)

    def detections(
        self,
        source_rgb: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> tuple[list[Detection], dict[str,int]]:
        rgb = np.asarray(source_rgb, dtype=np.uint8)
        sh, sw = rgb.shape[:2]
        raw: list[Detection] = []
        view_counts: dict[str,int] = {}

        for spec in _views():
            crop, box = _source_crop(rgb, spec)
            result = self._detect(crop)
            view_counts[spec.name] = len(result.hand_landmarks)
            for i, hand in enumerate(result.hand_landmarks):
                score = 1.0
                try:
                    if result.handedness and result.handedness[i]:
                        score = float(result.handedness[i][0].score)
                except Exception:
                    pass
                pts_source = _map_landmarks_to_source(hand, spec, box)
                pts_target = _source_to_target(
                    pts_source, sw, sh, target_width, target_height
                )
                raw.append(Detection(spec.name, score, pts_target))

        return _deduplicate(raw), view_counts
