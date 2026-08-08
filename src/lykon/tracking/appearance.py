"""Appearance features for stable player re-identification.

First version: HSV jersey / torso color histogram.
Architecture keeps a real ReID embedding interface for later (OSNet / torchreid).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import cv2
import numpy as np


class AppearanceExtractor(Protocol):
    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
        ...

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ...


def _clip_bbox(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def torso_roi(bbox_xyxy: list[float]) -> list[float]:
    """Upper-middle body crop used for jersey color."""
    x1, y1, x2, y2 = bbox_xyxy
    h = y2 - y1
    w = x2 - x1
    return [
        x1 + 0.2 * w,
        y1 + 0.15 * h,
        x2 - 0.2 * w,
        y1 + 0.55 * h,
    ]


class HSVHistogramAppearance:
    """Lightweight jersey-color embedding (HSV 2D histogram)."""

    def __init__(self, h_bins: int = 16, s_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins

    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        roi = torso_roi(bbox_xyxy)
        x1, y1, x2, y2 = _clip_bbox(roi, w, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            return 1.0
        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
        b = np.asarray(b, dtype=np.float32).reshape(-1, 1)
        # Bhattacharyya distance in [0, 1]
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

    def dominant_color_bgr(self, frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> list[int]:
        h, w = frame_bgr.shape[:2]
        roi = torso_roi(bbox_xyxy)
        x1, y1, x2, y2 = _clip_bbox(roi, w, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return [0, 0, 0]
        mean = crop.reshape(-1, 3).mean(axis=0)
        return [int(mean[0]), int(mean[1]), int(mean[2])]


class AppearanceReIDInterface:
    """Placeholder for heavier ReID models (OSNet / torchreid).

    Falls back to HSV histogram until a real model is wired in.
    """

    def __init__(self, backend: Optional[AppearanceExtractor] = None):
        self.backend: AppearanceExtractor = backend or HSVHistogramAppearance()

    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
        return self.backend.extract(frame_bgr, bbox_xyxy)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.backend.distance(a, b)


def color_histogram_distance(a: Any, b: Any) -> float:
    extractor = HSVHistogramAppearance()
    if a is None or b is None:
        return 1.0
    return extractor.distance(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
