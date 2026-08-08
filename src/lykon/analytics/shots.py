from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np

from lykon.schema import coerce_sample, player_id_of

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12


def _best_arm(kps: list[list[float]]) -> tuple[int, int, int]:
    left = np.mean([kps[i][2] for i in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)])
    right = np.mean([kps[i][2] for i in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)])
    return (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) if left >= right else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)


def detect_shot_motion_candidates(samples: list[dict[str, Any]], cooldown_s: float = 1.0) -> list[dict[str, Any]]:
    """Heuristic candidate generator only."""
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in samples:
        s = coerce_sample(raw)
        by_id[player_id_of(s)].append(s)

    events: list[dict[str, Any]] = []
    for pid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: r["time_s"])
        last_t = -1e9
        prev = None
        for s in rows:
            if not s.get("pose_valid", True):
                prev = s
                continue
            kps = s["keypoints"]
            sh, el, wr = _best_arm(kps)
            if min(kps[sh][2], kps[el][2], kps[wr][2]) < 0.35:
                prev = s
                continue
            wrist_above = kps[wr][1] < kps[sh][1]
            elbow_above_hip = kps[el][1] < np.mean([kps[LEFT_HIP][1], kps[RIGHT_HIP][1]])
            velocity = 0.0
            if prev is not None and prev.get("pose_valid", True):
                dt = s["time_s"] - prev["time_s"]
                if dt > 1e-6:
                    velocity = abs(kps[wr][1] - prev["keypoints"][wr][1]) / dt
            if wrist_above and elbow_above_hip and velocity > 80.0 and s["time_s"] - last_t >= cooldown_s:
                events.append({
                    "type": "shot_motion_candidate",
                    "stable_player_id": pid,
                    "track_id": pid,
                    "frame": s["frame_idx"],
                    "time_s": round(float(s["time_s"]), 3),
                    "confidence": "heuristic",
                    "note": "Requires ball/video semantics to confirm a real shot and outcome.",
                })
                last_t = s["time_s"]
            prev = s
    return events
