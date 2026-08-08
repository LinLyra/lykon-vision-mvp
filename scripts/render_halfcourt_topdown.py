#!/usr/bin/env python3
"""Project tracks_pose foot points onto a calibrated half court and render top-down.

Does NOT re-run YOLO. Reads:
  - tracks_pose.json (pixel_foot_point + stable_player_id)
  - halfcourt calibration JSON (image_points + world_points)

Writes:
  - tracks_pose_court.json  (same samples + court_xy_m)
  - halfcourt_topdown.mp4
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lykon.court.homography import compute_homography, image_to_court, load_court_config

PLAYER_COLORS = {
    "A": (255, 180, 40),   # BGR orange-ish
    "B": (40, 180, 255),   # BGR blue-ish
}
FALLBACK = [(80, 220, 120), (180, 80, 255), (40, 220, 220)]


def _pid(s: dict[str, Any]) -> str:
    return str(s.get("stable_player_id") or s.get("track_id") or "?")


def _frame(s: dict[str, Any]) -> int:
    return int(s.get("frame_idx", s.get("frame", 0)))


def _time(s: dict[str, Any]) -> float:
    if "time_s" in s:
        return float(s["time_s"])
    if "timestamp_s" in s:
        return float(s["timestamp_s"])
    return 0.0


def _foot(s: dict[str, Any]) -> tuple[float, float]:
    if s.get("pixel_foot_point") is not None:
        p = s["pixel_foot_point"]
        return float(p[0]), float(p[1])
    bbox = s.get("bbox_xyxy") or s.get("bbox")
    if bbox is None:
        raise KeyError("sample missing pixel_foot_point and bbox")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) * 0.5, y2


def court_size(cfg: dict) -> tuple[float, float]:
    size = cfg.get("court_size_m")
    if size and len(size) == 2:
        return float(size[0]), float(size[1])
    pts = cfg.get("world_points") or cfg.get("court_points_m") or []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(max(xs) - min(xs)) if xs else 15.0, float(max(ys) - min(ys)) if ys else 14.0


def project_tracks(samples: list[dict], H: np.ndarray) -> list[dict]:
    out = []
    for s in samples:
        item = dict(s)
        u, v = _foot(s)
        x, y = image_to_court((u, v), H)
        item["pixel_foot_point"] = [u, v]
        item["court_xy_m"] = [x, y]
        item["stable_player_id"] = _pid(s)
        out.append(item)
    return out


def draw_halfcourt(panel: np.ndarray, width_m: float, depth_m: float) -> None:
    """Simplified FIBA/NBA-ish half court markings in top-down panel."""
    h, w = panel.shape[:2]

    def to_px(xm: float, ym: float) -> tuple[int, int]:
        # baseline at bottom, midcourt at top
        px = int(np.clip(xm / width_m, 0, 1) * (w - 1))
        py = int(np.clip(1.0 - (ym / depth_m), 0, 1) * (h - 1))
        return px, py

    line = (210, 210, 220)
    dim = (120, 120, 130)

    # Outer boundary
    cv2.polylines(
        panel,
        [np.array([to_px(0, 0), to_px(width_m, 0), to_px(width_m, depth_m), to_px(0, depth_m)], dtype=np.int32)],
        True,
        line,
        2,
    )

    # Basket / rim projection on ground (~1.575 m from baseline, center)
    bx, by = width_m * 0.5, 1.575
    cv2.circle(panel, to_px(bx, by), 8, (60, 60, 220), 2)
    # backboard line
    cv2.line(panel, to_px(bx - 0.9, 1.2), to_px(bx + 0.9, 1.2), (90, 90, 200), 2)

    # Restricted area / paint (approx 4.9m wide, 5.8m deep)
    paint_w, paint_d = 4.9, 5.8
    x0 = (width_m - paint_w) * 0.5
    cv2.rectangle(panel, to_px(x0, 0), to_px(x0 + paint_w, paint_d), line, 1)

    # Free-throw circle (radius ~1.8m) centered at free-throw line
    ft_y = paint_d
    ft_r = 1.8
    # approximate ellipse in panel pixels
    c = to_px(bx, ft_y)
    rx = max(2, int((ft_r / width_m) * w))
    ry = max(2, int((ft_r / depth_m) * h))
    cv2.ellipse(panel, c, (rx, ry), 0, 0, 360, line, 1)

    # Three-point arc (approx): straight corners + arc radius ~6.75m from basket
    r3 = 6.75
    corner_x = 0.9  # sideline offset for NBA-like corners; simplified
    pts = []
    # left corner straight
    pts.append(to_px(corner_x, 0))
    pts.append(to_px(corner_x, 1.0))
    for deg in range(20, 161, 3):
        rad = math.radians(deg)
        # angles from +X; basket-centered arc opening toward midcourt (+Y)
        x = bx + r3 * math.cos(rad)
        y = by + r3 * math.sin(rad)
        if 0 <= x <= width_m and 0 <= y <= depth_m:
            pts.append(to_px(x, y))
    pts.append(to_px(width_m - corner_x, 1.0))
    pts.append(to_px(width_m - corner_x, 0))
    if len(pts) >= 2:
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, dim, 1)

    # Midcourt line
    cv2.line(panel, to_px(0, depth_m), to_px(width_m, depth_m), line, 2)

    # Center mark / hash
    cv2.line(panel, to_px(width_m * 0.5, depth_m - 0.3), to_px(width_m * 0.5, depth_m), dim, 1)


def render_topdown(
    samples: list[dict],
    output_mp4: Path,
    *,
    width_m: float,
    depth_m: float,
    fps: float,
    trail: int = 40,
    panel_size: tuple[int, int] = (720, 840),
) -> None:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        by_frame[_frame(s)].append(s)
    if not by_frame:
        raise ValueError("No samples to render")

    frames = sorted(by_frame)
    pw, ph = panel_size  # width, height
    # Prefer aspect matching court: width/depth
    ph = int(pw * (depth_m / width_m))
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    color_i = 0
    colors = dict(PLAYER_COLORS)

    def to_px(xm: float, ym: float) -> tuple[int, int]:
        px = int(np.clip(xm / width_m, 0, 1) * (pw - 1))
        py = int(np.clip(1.0 - (ym / depth_m), 0, 1) * (ph - 1))
        return px, py

    for fidx in range(frames[0], frames[-1] + 1):
        panel = np.full((ph, pw, 3), 32, dtype=np.uint8)
        draw_halfcourt(panel, width_m, depth_m)
        people = by_frame.get(fidx, [])
        for s in people:
            if "court_xy_m" not in s:
                continue
            pid = _pid(s)
            if pid not in colors:
                colors[pid] = FALLBACK[color_i % len(FALLBACK)]
                color_i += 1
            c = colors[pid]
            xy = s["court_xy_m"]
            history[pid].append((float(xy[0]), float(xy[1])))
            if len(history[pid]) > trail:
                history[pid] = history[pid][-trail:]
            pts = [to_px(x, y) for x, y in history[pid]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(panel, a, b, c, 2)
            cx, cy = to_px(xy[0], xy[1])
            cv2.circle(panel, (cx, cy), 11, c, -1)
            cv2.putText(panel, pid, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        cv2.putText(
            panel,
            "LYKON halfcourt top-down",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
        )
        writer.write(panel)

    writer.release()


def infer_fps(samples: list[dict]) -> float:
    times = sorted({_time(s) for s in samples})
    if len(times) >= 2:
        dts = np.diff(times)
        dts = dts[dts > 1e-6]
        if len(dts):
            return float(min(60.0, max(10.0, 1.0 / float(np.median(dts)))))
    return 30.0


def main():
    p = argparse.ArgumentParser(description="Map pose foot points to halfcourt XY and render top-down")
    p.add_argument("--pose-json", required=True, help="Existing tracks_pose.json")
    p.add_argument("--court-config", required=True, help="halfcourt_*.json from calibrate_halfcourt.py")
    p.add_argument("--output-dir", default="data/output/1v1_halfcourt")
    p.add_argument("--fps", type=float, default=None)
    args = p.parse_args()

    samples = json.loads(Path(args.pose_json).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("tracks_pose") or []
    if not samples:
        raise SystemExit(f"No samples in {args.pose_json}")

    cfg = load_court_config(args.court_config)
    H = compute_homography(cfg)
    width_m, depth_m = court_size(cfg)

    projected = project_tracks(samples, H)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    court_json = out_dir / "tracks_pose_court.json"
    court_json.write_text(json.dumps(projected, ensure_ascii=False, indent=2), encoding="utf-8")

    fps = float(args.fps) if args.fps else infer_fps(projected)
    mp4 = out_dir / "halfcourt_topdown.mp4"
    render_topdown(projected, mp4, width_m=width_m, depth_m=depth_m, fps=fps)

    players = sorted({_pid(s) for s in projected})
    print(
        json.dumps(
            {
                "players": players,
                "frames": len({_frame(s) for s in projected}),
                "court_size_m": [width_m, depth_m],
                "fps": fps,
                "outputs": {
                    "tracks_pose_court": str(court_json),
                    "halfcourt_topdown": str(mp4),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
