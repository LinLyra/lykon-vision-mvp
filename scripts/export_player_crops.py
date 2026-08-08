#!/usr/bin/env python3
"""Export stable-player crop videos for WHAM / SMPL input.

Reads original video + tracks_pose.json. Writes fixed-size crops:
  player_A.mp4 / player_B.mp4

Does not re-run YOLO.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _pid(s: dict[str, Any]) -> str:
    return str(s.get("stable_player_id") or s.get("track_id"))


def _frame(s: dict[str, Any]) -> int:
    return int(s.get("frame_idx", s.get("frame", 0)))


def _smooth_bbox_series(boxes: np.ndarray, window: int = 7) -> np.ndarray:
    """boxes: [T,4] xyxy. Light moving average to avoid crop jitter."""
    if len(boxes) < 3 or window <= 1:
        return boxes
    window = min(window, len(boxes))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    out = boxes.copy()
    for c in range(4):
        x = boxes[:, c]
        xp = np.pad(x, (pad, pad), mode="edge")
        out[:, c] = np.convolve(xp, kernel, mode="valid")
    return out


def _expand_square(bbox: list[float], margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    # keep full body preference: use max side after margin
    half = 0.5 * side
    nx1 = int(round(cx - half))
    ny1 = int(round(cy - half))
    nx2 = int(round(cx + half))
    ny2 = int(round(cy + half))
    # shift into frame if possible without shrinking first
    if nx2 - nx1 < 2:
        nx2 = nx1 + 2
    if ny2 - ny1 < 2:
        ny2 = ny1 + 2
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > w:
        shift = nx2 - w
        nx1 = max(0, nx1 - shift)
        nx2 = w
    if ny2 > h:
        shift = ny2 - h
        ny1 = max(0, ny1 - shift)
        ny2 = h
    return nx1, ny1, nx2, ny2


def export_player_crop(
    video_path: str,
    samples: list[dict[str, Any]],
    player_id: str,
    output_mp4: Path,
    *,
    size: int = 640,
    margin: float = 0.25,
) -> dict[str, Any]:
    rows = [s for s in samples if _pid(s) == player_id]
    if not rows:
        raise SystemExit(f"No samples for player {player_id}")
    rows = sorted(rows, key=_frame)

    by_frame = {_frame(s): s for s in rows}
    f0, f1 = min(by_frame), max(by_frame)

    # Build dense bbox series with hold/predict for gaps
    frame_ids = list(range(f0, f1 + 1))
    boxes = []
    confs = []
    last = None
    vel = np.zeros(4, dtype=float)
    for f in frame_ids:
        s = by_frame.get(f)
        if s is not None and s.get("bbox_xyxy"):
            box = np.asarray(s["bbox_xyxy"], dtype=float)
            if last is not None:
                vel = 0.7 * vel + 0.3 * (box - last)
            last = box
            boxes.append(box)
            confs.append(1.0)
        elif last is not None:
            pred = last + vel
            boxes.append(pred)
            last = pred
            confs.append(0.0)
        else:
            boxes.append(np.array([0, 0, size, size], dtype=float))
            confs.append(0.0)
    boxes_arr = _smooth_bbox_series(np.asarray(boxes, dtype=float), window=7)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size, size),
    )

    written = 0
    for i, f in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            # blank fallback to keep length
            crop = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            x1, y1, x2, y2 = _expand_square(boxes_arr[i].tolist(), margin, vw, vh)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                crop = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
        writer.write(crop)
        written += 1

    cap.release()
    writer.release()
    return {
        "player_id": player_id,
        "frames": written,
        "fps": fps,
        "size": size,
        "margin": margin,
        "output": str(output_mp4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export player A/B crop videos for 3D reconstruction")
    p.add_argument("--video", required=True)
    p.add_argument("--pose-json", required=True)
    p.add_argument("--output-dir", default="data/output/1v1_36s")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--margin", type=float, default=0.25, help="Extra margin around bbox (0.25 = 25%)")
    p.add_argument("--players", default="A,B")
    args = p.parse_args()

    samples = json.loads(Path(args.pose_json).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("tracks_pose") or []

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pid in [x.strip() for x in args.players.split(",") if x.strip()]:
        path = out_dir / f"player_{pid}.mp4"
        results.append(
            export_player_crop(
                args.video,
                samples,
                pid,
                path,
                size=args.size,
                margin=args.margin,
            )
        )
    print(json.dumps({"crops": results}, indent=2))


if __name__ == "__main__":
    main()
