"""Court ROI polygon filtering for active-player selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


def load_court_roi(path: str | Path | dict | None) -> dict | None:
    if path is None:
        return None
    if isinstance(path, dict):
        return path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_polygon(roi_config: dict | None) -> np.ndarray | None:
    if not roi_config:
        return None
    poly = roi_config.get("court_polygon_pixels") or roi_config.get("polygon")
    if not poly:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("court_polygon_pixels must be a list of >=3 [x,y] points")
    return arr


def point_in_polygon(point_xy: Sequence[float], polygon: np.ndarray) -> bool:
    """Return True if point lies inside / on the polygon."""
    x, y = float(point_xy[0]), float(point_xy[1])
    # cv2.pointPolygonTest: >0 inside, 0 edge, <0 outside
    return cv2.pointPolygonTest(polygon.astype(np.float32), (x, y), False) >= 0


def foot_point_from_bbox(bbox_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [(x1 + x2) / 2.0, y2]


def is_on_court(
    foot_xy: Sequence[float],
    roi_config: dict | None,
    *,
    default_if_missing: bool = True,
) -> bool:
    poly = get_polygon(roi_config)
    if poly is None:
        return default_if_missing
    return point_in_polygon(foot_xy, poly)


def filter_detections_by_roi(
    detections: list[dict[str, Any]],
    roi_config: dict | None,
    *,
    foot_key: str = "pixel_foot_point",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split detections into (active_on_court, ignored_off_court)."""
    if get_polygon(roi_config) is None:
        return detections, []

    active: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for det in detections:
        foot = det.get(foot_key)
        if foot is None:
            bbox = det.get("bbox_xyxy") or det.get("bbox")
            foot = foot_point_from_bbox(bbox) if bbox is not None else [0.0, 0.0]
        if is_on_court(foot, roi_config):
            active.append(det)
        else:
            ignored.append(det)
    return active, ignored


def save_court_roi(polygon_pixels: list[list[float]], output_path: str | Path, meta: dict | None = None) -> dict:
    payload = {
        "court_polygon_pixels": [[float(x), float(y)] for x, y in polygon_pixels],
        "note": "Persons whose foot/bottom-center falls outside this polygon are ignored for active player tracking.",
    }
    if meta:
        payload.update(meta)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def draw_roi(frame_bgr: np.ndarray, roi_config: dict | None, color=(0, 200, 255), thickness: int = 2) -> np.ndarray:
    poly = get_polygon(roi_config)
    if poly is None:
        return frame_bgr
    out = frame_bgr.copy()
    pts = poly.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
