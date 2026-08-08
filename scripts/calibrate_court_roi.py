#!/usr/bin/env python3
"""Click a polygon around the playable court ROI to filter spectators/coaches."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2

from lykon.court.roi import save_court_roi


def main():
    p = argparse.ArgumentParser(description="Calibrate court ROI polygon")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame", type=int, default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(args.frame, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame")

    points: list[tuple[int, int]] = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 6, (0, 200, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if len(points) >= 2:
            for a, b in zip(points, points[1:]):
                cv2.line(display, a, b, (0, 200, 255), 2)
        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 200, 255), 1)
        cv2.putText(
            display,
            "Click court polygon (>=3). ENTER=save  u=undo  ESC=cancel",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    window = "Lykon Court ROI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key in (13, 10):  # Enter
            if len(points) < 3:
                continue
            break

    cv2.destroyAllWindows()
    poly = [[float(x), float(y)] for x, y in points]
    cfg = save_court_roi(poly, args.output, meta={"init_frame": frame_idx, "video": str(args.video)})
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
