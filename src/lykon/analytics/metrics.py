from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np


def _smooth_xy(points: np.ndarray, window: int = 5) -> np.ndarray:
    if len(points) < 3 or window <= 1:
        return points
    window = min(window, len(points))
    kernel = np.ones(window) / window
    x = np.convolve(points[:, 0], kernel, mode="same")
    y = np.convolve(points[:, 1], kernel, mode="same")
    x[: window // 2] = points[: window // 2, 0]
    y[: window // 2] = points[: window // 2, 1]
    x[-window // 2 :] = points[-window // 2 :, 0]
    y[-window // 2 :] = points[-window // 2 :, 1]
    return np.column_stack([x, y])


def _pid(row: dict[str, Any]) -> str:
    return str(row.get("stable_player_id", row.get("track_id")))


def compute_motion_metrics(trajectories: list[dict[str, Any]], smoothing_window: int = 5) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trajectories:
        if "court_xy_m" in t:
            by_id[_pid(t)].append(t)

    players: dict[str, Any] = {}
    for pid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: r["time_s"])
        times = np.asarray([r["time_s"] for r in rows], dtype=float)
        xy = np.asarray([r["court_xy_m"] for r in rows], dtype=float)
        xy = _smooth_xy(xy, smoothing_window)
        dt = np.diff(times)
        dxy = np.diff(xy, axis=0)
        step = np.linalg.norm(dxy, axis=1)
        valid = dt > 1e-6
        speeds = np.zeros_like(step)
        speeds[valid] = step[valid] / dt[valid]
        clean = speeds[(speeds >= 0) & (speeds <= 12.0)]
        distance = float(step[speeds <= 12.0].sum()) if len(step) else 0.0
        players[str(pid)] = {
            "distance_m": round(distance, 2),
            "avg_speed_mps": round(float(clean.mean()) if len(clean) else 0.0, 2),
            "max_speed_mps": round(float(np.percentile(clean, 95)) if len(clean) else 0.0, 2),
            "samples": len(rows),
        }

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in trajectories:
        if "court_xy_m" in t:
            by_frame[int(t["frame"])].append(t)
    separations = []
    for rows in by_frame.values():
        if len(rows) >= 2:
            rows = sorted(rows, key=_pid)[:2]
            a = np.asarray(rows[0]["court_xy_m"], float)
            b = np.asarray(rows[1]["court_xy_m"], float)
            separations.append(float(np.linalg.norm(a - b)))
    spacing = {
        "avg_1v1_separation_m": round(float(np.mean(separations)), 2) if separations else None,
        "min_1v1_separation_m": round(float(np.percentile(separations, 10)), 2) if separations else None,
    }
    return {"players": players, "spacing": spacing}
