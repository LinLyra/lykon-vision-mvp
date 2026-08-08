from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def load_court_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_homography(config: dict) -> np.ndarray:
    src = np.asarray(config["image_points"], dtype=np.float32)
    dst_key = "world_points" if "world_points" in config else "court_points_m"
    dst = np.asarray(config[dst_key], dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Court config requires four image_points and four world/court points")
    # Prefer findHomography for robustness; fall back to getPerspectiveTransform
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        H = cv2.getPerspectiveTransform(src, dst)
    return H


def image_to_court(point_xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    p = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])
