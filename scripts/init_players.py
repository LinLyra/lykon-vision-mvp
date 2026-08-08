#!/usr/bin/env python3
"""Human-initialized player identity setup for Lykon Vision.

Click players in order. Press ENTER after naming each player (CLI prompts).
Keys:
  left-click  — select bbox corner / center click for quick box
  n           — next player after current selection
  u           — undo last point
  q / ESC     — cancel
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lykon.tracking.appearance import HSVHistogramAppearance


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
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame, frame_idx


def _default_names(num_players: int, mode_hint: str | None) -> list[str]:
    if num_players == 2:
        return ["A", "B"]
    if num_players == 6:
        return ["A1", "A2", "A3", "B1", "B2", "B3"]
    names = []
    for i in range(num_players):
        team = "A" if i < (num_players + 1) // 2 else "B"
        names.append(f"{team}{i+1}")
    return names


def main():
    p = argparse.ArgumentParser(description="Initialize stable Lykon player identities")
    p.add_argument("--video", required=True)
    p.add_argument("--num-players", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frame", type=int, default=0, help="Init from this frame if first frame is occluded")
    p.add_argument("--init-frame", type=int, default=None, dest="init_frame", help="Alias for --frame")
    p.add_argument("--names", default=None, help="Comma-separated player ids, e.g. A,B")
    args = p.parse_args()
    if args.init_frame is not None:
        args.frame = args.init_frame

    frame, frame_idx = _read_frame(args.video, args.frame)
    names = [x.strip() for x in args.names.split(",")] if args.names else _default_names(args.num_players, None)
    if len(names) != args.num_players:
        raise SystemExit(f"--names length ({len(names)}) must equal --num-players ({args.num_players})")

    appearance = HSVHistogramAppearance()
    players = []
    display = frame.copy()
    points: list[tuple[int, int]] = []
    current_idx = 0

    def redraw():
        nonlocal display
        display = frame.copy()
        if current_idx < len(names):
            hint = (
                f"Init player {names[current_idx]} ({current_idx+1}/{len(names)}) "
                "— click 2 corners of bbox"
            )
        else:
            hint = "Done — saving player init..."
        cv2.putText(
            display,
            hint,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        for pl in players:
            x1, y1, x2, y2 = map(int, pl["bbox"])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.putText(display, pl["player_id"], (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
        for pt in points:
            cv2.circle(display, pt, 5, (0, 255, 255), -1)

    def on_mouse(event, x, y, flags, param):
        nonlocal points, current_idx, players
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if current_idx >= len(names):
            return
        points.append((x, y))
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            bbox = [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
            # Expand tiny clicks
            if bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10:
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                bbox = [cx - 40, cy - 80, cx + 40, cy + 80]
            pid = names[current_idx]
            team = pid[0] if pid and pid[0] in ("A", "B") else ""
            feat = appearance.extract(frame, bbox)
            jersey = appearance.dominant_color_bgr(frame, bbox)
            players.append({
                "player_id": pid,
                "team": team,
                "bbox": bbox,
                "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                "appearance": feat.tolist(),
                "jersey_color": jersey,
                "init_frame": frame_idx,
            })
            points = []
            current_idx += 1
        redraw()

    window = "Lykon Player Init"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while current_idx < len(names):
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")
        if key == ord("u") and players:
            players.pop()
            current_idx = max(0, current_idx - 1)
            points = []
            redraw()

    cv2.destroyAllWindows()
    payload = {
        "video": str(args.video),
        "init_frame": frame_idx,
        "players": players,
        "note": "Human-initialized stable player identities for Lykon Vision.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), "players": [p["player_id"] for p in players]}, indent=2))


if __name__ == "__main__":
    main()
