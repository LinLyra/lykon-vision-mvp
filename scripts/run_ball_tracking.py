#!/usr/bin/env python3
"""Run sports-ball detection + temporal tracking (no pose / no person re-run)."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2

from lykon.ball.detector import BallDetector
from lykon.ball.tracker import BallTracker
from lykon.video.io import probe_video


def main() -> None:
    p = argparse.ArgumentParser(description="Track basketball (COCO sports ball) in a fixed-camera clip")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="data/output/1v1_36s/ball_track.json")
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-lost-frames", type=int, default=20)
    p.add_argument("--max-displacement-px", type=float, default=180.0)
    p.add_argument("--interpolate-gap", type=int, default=8)
    p.add_argument("--device", default=None, help="Optional ultralytics device, e.g. cpu / mps / 0")
    args = p.parse_args()

    info = probe_video(args.video)
    detector = BallDetector(
        args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    tracker = BallTracker(
        max_lost_frames=args.max_lost_frames,
        max_displacement_px=args.max_displacement_px,
        interpolate_gap=args.interpolate_gap,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frames = []
    frame_idx = 0
    detected_n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / info.fps if info.fps > 0 else 0.0
        dets = detector.detect(frame)
        ball = tracker.update(frame_idx, time_s, dets)
        if ball.detected:
            detected_n += 1
        frames.append(ball.to_dict())
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"ball tracking frame {frame_idx}/{info.frame_count}", flush=True)

    cap.release()

    payload = {
        "video": str(args.video),
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration_s": info.duration_s,
        "params": {
            "model": args.model,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "sports_ball_class": 32,
            "max_lost_frames": args.max_lost_frames,
            "max_displacement_px": args.max_displacement_px,
            "interpolate_gap": args.interpolate_gap,
        },
        "stats": {
            "frames": len(frames),
            "detected_frames": detected_n,
            "detection_coverage": round(detected_n / max(len(frames), 1), 4),
            "predicted_or_interpolated": sum(1 for f in frames if f["source"] != "detected"),
        },
        "frames": frames,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), "stats": payload["stats"]}, indent=2))


if __name__ == "__main__":
    main()
