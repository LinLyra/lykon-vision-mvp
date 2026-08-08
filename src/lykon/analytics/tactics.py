from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np


def _pid(row: dict[str, Any]) -> str:
    return str(row.get("stable_player_id", row.get("track_id")))


def summarize_1v1_tactics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Geometry-only 1v1 tactical summary."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if "court_xy_m" in row:
            by_frame[int(row["frame"])].append(row)

    close_windows = []
    current = None
    for frame in sorted(by_frame):
        rows = by_frame[frame]
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=_pid)[:2]
        a, b = np.asarray(rows[0]["court_xy_m"]), np.asarray(rows[1]["court_xy_m"])
        d = float(np.linalg.norm(a - b))
        t = float(rows[0]["time_s"])
        if d <= 1.5:
            if current is None:
                current = {"start_s": t, "end_s": t, "min_distance_m": d}
            else:
                current["end_s"] = t
                current["min_distance_m"] = min(current["min_distance_m"], d)
        elif current is not None:
            if current["end_s"] - current["start_s"] >= 0.2:
                close_windows.append(current)
            current = None
    if current is not None:
        close_windows.append(current)

    return {
        "close_contact_windows": [
            {
                "start_s": round(x["start_s"], 2),
                "end_s": round(x["end_s"], 2),
                "min_distance_m": round(x["min_distance_m"], 2),
            }
            for x in close_windows
        ],
        "interpretation": "Geometry-only. Add ball/possession semantics before naming drives, closeouts, screens or shot contests.",
    }
