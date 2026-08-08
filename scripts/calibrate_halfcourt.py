#!/usr/bin/env python3
"""Manual half-court calibration for a fixed basketball camera.

Click 4 known ground points on the first (or chosen) frame, mapped to a
standard half court: width=15m, depth=14m (baseline → midcourt).

Recommended click order (default world_points):
  1. left baseline × sideline     → (0, 0)
  2. right baseline × sideline    → (15, 0)
  3. right midcourt × sideline    → (15, 14)
  4. left midcourt × sideline     → (0, 14)

If midcourt is not visible, use any 4 known points and pass --world-points.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_WORLD = [
    [0.0, 0.0],
    [15.0, 0.0],
    [15.0, 14.0],
    [0.0, 14.0],
]

LABELS = [
    "1 left baseline x sideline",
    "2 right baseline x sideline",
    "3 right midcourt x sideline",
    "4 left midcourt x sideline",
]


def _read_frame(video_path: str, frame_idx: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame, frame_idx


def calibrate_halfcourt(
    video_path: str,
    output_path: str,
    *,
    width_m: float = 15.0,
    depth_m: float = 14.0,
    frame_idx: int = 0,
    world_points: list[list[float]] | None = None,
) -> dict:
    frame, used_frame = _read_frame(video_path, frame_idx)
    world = world_points or [
        [0.0, 0.0],
        [width_m, 0.0],
        [width_m, depth_m],
        [0.0, depth_m],
    ]
    if len(world) != 4:
        raise ValueError("world_points must contain exactly 4 [X,Y] points")

    points: list[tuple[int, int]] = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        hint = LABELS[len(points)] if len(points) < 4 else "ENTER=save  u=undo  ESC=cancel"
        cv2.putText(
            display,
            f"Halfcourt calibrate: click {hint}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"world target: {world[len(points)] if len(points) < 4 else 'done'}  frame={used_frame}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
        )
        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(points) >= 2:
            for a, b in zip(points, points[1:]):
                cv2.line(display, a, b, (0, 200, 255), 2)
        if len(points) == 4:
            cv2.line(display, points[-1], points[0], (0, 200, 255), 2)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    window = "Lykon Halfcourt Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled")
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key in (13, 10) and len(points) == 4:
            break
        if len(points) == 4 and key == ord("s"):
            break

    cv2.destroyAllWindows()

    config = {
        "court_type": "halfcourt",
        "court_size_m": [float(width_m), float(depth_m)],
        "image_points": [[float(x), float(y)] for x, y in points],
        "world_points": [[float(x), float(y)] for x, y in world],
        # backward-compatible alias used by older modules
        "court_points_m": [[float(x), float(y)] for x, y in world],
        "video": str(video_path),
        "init_frame": used_frame,
        "click_order_note": (
            "Default: 1 left-baseline, 2 right-baseline, 3 right-midcourt, 4 left-midcourt. "
            "Origin (0,0)=left baseline corner; +X along baseline; +Y toward midcourt."
        ),
        "basket_note": "Basket center approx (7.5, 1.575) m on ground; rim height 3.05 m (unused in 2D).",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main():
    p = argparse.ArgumentParser(description="Calibrate fixed-camera half court via 4 ground points")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width-m", type=float, default=15.0)
    p.add_argument("--depth-m", type=float, default=14.0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument(
        "--world-points",
        default=None,
        help="Optional override: 8 numbers x1,y1,...,x4,y4 in meters",
    )
    args = p.parse_args()

    world = None
    if args.world_points:
        vals = [float(x) for x in args.world_points.replace(",", " ").split()]
        if len(vals) != 8:
            raise SystemExit("--world-points needs 8 numbers")
        world = [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]

    cfg = calibrate_halfcourt(
        args.video,
        args.output,
        width_m=args.width_m,
        depth_m=args.depth_m,
        frame_idx=args.frame,
        world_points=world,
    )
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
