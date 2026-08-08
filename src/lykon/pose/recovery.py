"""Short-gap pose recovery via interpolation. Never invent high-confidence pose."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np

from lykon.schema import NUM_KEYPOINTS, coerce_sample, normalize_keypoints, pose_valid_from_keypoints

InterpKind = Literal["linear", "cubic"]


def _interp_series(xs: np.ndarray, ys: np.ndarray, x_query: np.ndarray, kind: InterpKind) -> np.ndarray:
    if len(xs) == 1:
        return np.full_like(x_query, ys[0], dtype=float)
    if kind == "cubic" and len(xs) >= 4:
        try:
            from scipy.interpolate import CubicSpline

            cs = CubicSpline(xs, ys, extrapolate=False)
            out = cs(x_query)
            return np.asarray(out, dtype=float)
        except Exception:
            pass
    return np.interp(x_query, xs, ys).astype(float)


def recover_poses(
    samples: list[dict[str, Any]],
    *,
    max_gap: int = 8,
    kind: InterpKind = "linear",
    conf_floor: float = 0.15,
) -> list[dict[str, Any]]:
    """Fill short pose gaps per stable player.

    Rules:
    - Gaps shorter than max_gap frames are interpolated.
    - Interpolated keypoints get reduced confidence (never fake high conf).
    - Longer gaps remain pose_source=missing / pose_valid=false.
    Identity / bbox continuity is preserved regardless of pose.
    """
    if not samples:
        return []

    normalized = [coerce_sample(s) for s in samples]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in normalized:
        by_player[str(s["stable_player_id"])].append(s)

    recovered: list[dict[str, Any]] = []
    for pid, rows in by_player.items():
        rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
        frames = [int(r["frame_idx"]) for r in rows]
        frame_set = set(frames)
        if not frames:
            continue

        # Index existing detected poses
        detected = {
            int(r["frame_idx"]): r
            for r in rows
            if r.get("pose_source") != "missing" and pose_valid_from_keypoints(r["keypoints"])
        }

        f0, f1 = frames[0], frames[-1]
        # Build dense timeline only for frames where player appears (or short gaps between appearances)
        appearance_frames = sorted(frame_set)
        # Also include interpolated missing frames between consecutive observations if gap small
        dense_frames = set(appearance_frames)
        for a, b in zip(appearance_frames, appearance_frames[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for f in range(a + 1, b):
                    dense_frames.add(f)

        det_frames = sorted(detected.keys())
        for f in sorted(dense_frames):
            if f in detected:
                s = dict(detected[f])
                s["pose_source"] = "detected"
                s["pose_valid"] = True
                recovered.append(s)
                continue

            # Find neighbors for interpolation
            left = [d for d in det_frames if d < f]
            right = [d for d in det_frames if d > f]
            if not left or not right:
                # Keep identity row if present without inventing pose
                base = next((r for r in rows if int(r["frame_idx"]) == f), None)
                if base is not None:
                    s = dict(base)
                    s["keypoints"] = normalize_keypoints(None)
                    s["pose_valid"] = False
                    s["pose_source"] = "missing"
                    recovered.append(s)
                continue

            fl, fr = left[-1], right[0]
            gap = fr - fl - 1
            if gap > max_gap or not (fl < f < fr):
                base = next((r for r in rows if int(r["frame_idx"]) == f), None)
                if base is not None:
                    s = dict(base)
                    s["keypoints"] = normalize_keypoints(None)
                    s["pose_valid"] = False
                    s["pose_source"] = "missing"
                    recovered.append(s)
                continue

            left_s = detected[fl]
            right_s = detected[fr]
            alpha = (f - fl) / float(fr - fl)
            kps = []
            for i in range(NUM_KEYPOINTS):
                lx, ly, lc = left_s["keypoints"][i]
                rx, ry, rc = right_s["keypoints"][i]
                if lc < 0.1 and rc < 0.1:
                    kps.append([0.0, 0.0, 0.0])
                    continue
                # Linear blend; optional cubic uses neighbor series if enough history
                xs = np.array([fl, fr], dtype=float)
                xq = np.array([float(f)])
                x = float(_interp_series(xs, np.array([lx, rx]), xq, kind)[0])
                y = float(_interp_series(xs, np.array([ly, ry]), xq, kind)[0])
                c = float(min(lc, rc) * (1.0 - abs(0.5 - alpha) * 0.5))
                c = min(c, conf_floor + 0.25)  # never high-confidence forged pose
                kps.append([x, y, c])

            # BBox interpolate from neighbors
            lb = left_s["bbox_xyxy"]
            rb = right_s["bbox_xyxy"]
            bbox = [lb[i] * (1 - alpha) + rb[i] * alpha for i in range(4)]
            time_s = left_s["time_s"] * (1 - alpha) + right_s["time_s"] * alpha
            s = {
                "frame_idx": f,
                "frame": f,
                "time_s": time_s,
                "timestamp_s": time_s,
                "stable_player_id": pid,
                "track_id": pid,
                "temporary_track_id": left_s.get("temporary_track_id"),
                "bbox_xyxy": bbox,
                "bbox": bbox,
                "detection_confidence": 0.0,
                "confidence": 0.0,
                "keypoints": kps,
                "pose_valid": pose_valid_from_keypoints(kps),
                "pose_source": "interpolated",
                "pixel_foot_point": [
                    (bbox[0] + bbox[2]) / 2.0,
                    bbox[3],
                ],
                "tracking_state": "occluded",
                "team": left_s.get("team"),
            }
            if "court_xy_m" in left_s and "court_xy_m" in right_s:
                s["court_xy_m"] = [
                    left_s["court_xy_m"][0] * (1 - alpha) + right_s["court_xy_m"][0] * alpha,
                    left_s["court_xy_m"][1] * (1 - alpha) + right_s["court_xy_m"][1] * alpha,
                ]
            recovered.append(s)

    recovered.sort(key=lambda r: (int(r["frame_idx"]), str(r["stable_player_id"])))
    return recovered
