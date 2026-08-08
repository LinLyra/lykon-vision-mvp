from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import numpy as np

from lykon.schema import coerce_sample, player_id_of

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
]


def _normalize_pose(kps: list[list[float]], depth_scale: float = 0.35) -> list[list[float]]:
    arr = np.asarray(kps, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return [[0.0, 0.0, 0.0, 0.0] for _ in range(17)]
    xy = arr[:, :2]
    conf = arr[:, 2]
    valid = conf > 0.2
    if valid.sum() < 4:
        return [[0.0, 0.0, 0.0, float(c)] for c in conf]

    if conf[11] > 0.2 and conf[12] > 0.2:
        center = (xy[11] + xy[12]) / 2.0
        torso = np.linalg.norm(((xy[5] + xy[6]) / 2.0) - center)
    else:
        center = xy[valid].mean(axis=0)
        torso = np.ptp(xy[valid, 1]) * 0.25
    scale = max(float(torso), 30.0)

    x = (xy[:, 0] - center[0]) / scale
    y = -(xy[:, 1] - center[1]) / scale

    z = np.zeros(len(arr), dtype=float)
    left = [1, 3, 5, 7, 9, 11, 13, 15]
    right = [2, 4, 6, 8, 10, 12, 14, 16]
    z[left] -= depth_scale * 0.08
    z[right] += depth_scale * 0.08
    z += depth_scale * 0.12 * x

    return [[float(xi), float(yi), float(zi), float(ci)] for xi, yi, zi, ci in zip(x, y, z, conf)]


def build_pseudo3d(samples: list[dict[str, Any]], trajectories: list[dict[str, Any]] | None = None, depth_scale: float = 0.35) -> dict:
    traj_lookup = {}
    if trajectories:
        for t in trajectories:
            tid = str(t.get("stable_player_id", t.get("track_id")))
            traj_lookup[(int(t["frame"]), tid)] = t

    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in samples:
        s = coerce_sample(raw)
        if not s.get("pose_valid", True) and s.get("pose_source") == "missing":
            continue
        pid = player_id_of(s)
        body = _normalize_pose(s["keypoints"], depth_scale=depth_scale)
        t = traj_lookup.get((int(s["frame_idx"]), pid))
        root = t.get("court_xy_m") if t and "court_xy_m" in t else [0.0, 0.0]
        frames[int(s["frame_idx"])].append({
            "stable_player_id": pid,
            "track_id": pid,
            "time_s": float(s["time_s"]),
            "root_xy": root,
            "joints_xyz_conf": body,
        })

    return {
        "note": "Pseudo-3D is a fast visualization baseline, not metric 3D pose. Upgrade to WHAM/SMPL for high-fidelity reconstruction.",
        "skeleton_edges": SKELETON_EDGES,
        "frames": {str(k): v for k, v in sorted(frames.items())},
    }


def save_pseudo3d(payload: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
