"""Overlay and top-down court visualization."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lykon.court.roi import draw_roi
from lykon.schema import player_id_of


TEAM_COLORS = {
    "A": (40, 180, 255),
    "B": (255, 160, 40),
}


def _team(pid: str, sample: dict | None = None) -> str:
    if sample and sample.get("team"):
        return str(sample["team"])
    return pid[0] if pid and pid[0] in ("A", "B") else ""


def render_tracked_overlay(
    video_path: str | Path,
    samples: list[dict[str, Any]],
    output_path: str | Path,
    *,
    court_roi: dict | None = None,
    debug: bool = False,
) -> Path:
    """Render overlay video labeled by stable_player_id."""
    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_frame[int(s.get("frame_idx", s.get("frame", 0)))].append(s)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw_roi(frame, court_roi)
        for s in by_frame.get(frame_idx, []):
            pid = player_id_of(s)
            color = TEAM_COLORS.get(_team(pid, s), (0, 220, 120))
            x1, y1, x2, y2 = map(int, s["bbox_xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if debug:
                label = (
                    f"{pid} tmp={s.get('temporary_track_id')} "
                    f"{s.get('tracking_state')} {s.get('pose_source')}"
                )
                cv2.putText(frame, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.putText(frame, pid, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                tmp = s.get("temporary_track_id")
                if tmp is not None:
                    cv2.putText(frame, f"t{tmp}", (x1, min(h - 6, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            for kp in s.get("keypoints") or []:
                if len(kp) >= 3 and kp[2] >= 0.2 and kp[0] > 0 and kp[1] > 0:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 180, 255), -1)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


def render_court_topdown(
    samples: list[dict[str, Any]],
    video_path: str | Path,
    output_path: str | Path,
    *,
    court_config: dict | None = None,
    width_m: float = 15.0,
    depth_m: float = 14.0,
    trail: int = 45,
) -> Path | None:
    """Side-by-side original video + top-down court trajectories.

    Returns None if no court_xy_m samples are available.
    """
    has_court = any("court_xy_m" in s for s in samples)
    if not has_court:
        return None

    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        by_frame[int(s.get("frame_idx", s.get("frame", 0)))].append(s)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_h = vh
    panel_w = int(panel_h * (width_m / depth_m)) if depth_m > 0 else vh
    out_w = vw + panel_w
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, panel_h))

    def court_to_px(xy: list[float]) -> tuple[int, int]:
        x_m, y_m = xy
        px = int(np.clip(x_m / width_m, 0, 1) * (panel_w - 1))
        # near sideline at bottom of panel
        py = int(np.clip(1.0 - (y_m / depth_m), 0, 1) * (panel_h - 1))
        return px, py

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panel = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)
        # court outline
        cv2.rectangle(panel, (10, 10), (panel_w - 10, panel_h - 10), (70, 70, 80), 2)
        cv2.line(panel, (panel_w // 2, 10), (panel_w // 2, panel_h - 10), (50, 50, 60), 1)
        # free-throw-ish arc placeholder
        cv2.circle(panel, (panel_w // 2, int(panel_h * 0.15)), int(panel_w * 0.12), (50, 50, 60), 1)

        for s in by_frame.get(frame_idx, []):
            if "court_xy_m" not in s:
                continue
            pid = player_id_of(s)
            color = TEAM_COLORS.get(_team(pid, s), (0, 220, 120))
            xy = s["court_xy_m"]
            history[pid].append((float(xy[0]), float(xy[1])))
            if len(history[pid]) > trail:
                history[pid] = history[pid][-trail:]
            pts = [court_to_px(list(p)) for p in history[pid]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(panel, a, b, color, 2)
            cx, cy = court_to_px(xy)
            cv2.circle(panel, (cx, cy), 10, color, -1)
            cv2.putText(panel, pid, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # left: video, right: topdown
        if frame.shape[0] != panel_h or frame.shape[1] != vw:
            frame = cv2.resize(frame, (vw, panel_h))
        canvas = np.concatenate([frame, panel], axis=1)
        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
