#!/usr/bin/env python3
"""Manually initialize hoop anchor for a fixed-camera clip.

Left-click: set hoop center.
Optional drag / two clicks: rim-ish bbox.
Keys:
  ENTER / s  — save
  u          — undo
  b          — toggle bbox mode (click 2 corners)
  ESC / q    — cancel
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2


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


def main() -> None:
    p = argparse.ArgumentParser(description="Click hoop center (fixed camera anchor)")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="configs/hoop_1v1_36s.json")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--init-frame", type=int, default=None, dest="init_frame")
    args = p.parse_args()
    if args.init_frame is not None:
        args.frame = args.init_frame

    frame, frame_idx = _read_frame(args.video, args.frame)
    center: list[float] | None = None
    bbox_points: list[tuple[int, int]] = []
    bbox: list[float] | None = None
    bbox_mode = False
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        mode = "BBOX: click 2 corners of rim/backboard" if bbox_mode else "CENTER: click hoop center"
        cv2.putText(
            display,
            f"{mode} | ENTER=save  b=toggle bbox  u=undo  ESC=cancel  frame={frame_idx}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
            cv2.circle(display, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(display, "HOOP", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for pt in bbox_points:
            cv2.circle(display, pt, 5, (0, 200, 255), -1)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)

    def on_mouse(event, x, y, flags, param):
        nonlocal center, bbox_points, bbox
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if bbox_mode:
            bbox_points.append((x, y))
            if len(bbox_points) >= 2:
                (x1, y1), (x2, y2) = bbox_points[-2], bbox_points[-1]
                bbox = [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
                bbox_points = bbox_points[-2:]
                if center is None:
                    center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
        else:
            center = [float(x), float(y)]
        redraw()

    window = "Lykon Hoop Init"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")
        if key == ord("b"):
            bbox_mode = not bbox_mode
            redraw()
        if key == ord("u"):
            if bbox_mode and bbox_points:
                bbox_points.pop()
                bbox = None
            else:
                center = None
            redraw()
        if key in (13, 10, ord("s")):
            if center is None:
                continue
            break

    cv2.destroyAllWindows()
    payload = {
        "video": str(args.video),
        "frame_idx": int(frame_idx),
        "hoop_center_pixel": [float(center[0]), float(center[1])],
        "note": "Fixed-camera hoop anchor. Used for all frames until camera moves.",
    }
    if bbox is not None:
        payload["hoop_bbox_xyxy"] = bbox
        payload["rim_or_backboard_bbox_xyxy"] = bbox

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), **payload}, indent=2))


if __name__ == "__main__":
    main()
