#!/usr/bin/env python3
"""Overlay players + ball + hoop, and write ball-player relation features.

Does not re-run YOLO person tracking. Reads existing tracks_pose.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# COCO wrists
LEFT_WRIST, RIGHT_WRIST = 9, 10

PLAYER_COLORS = {
    "A": (255, 160, 40),  # BGR: blue-ish for A
    "B": (40, 180, 255),  # BGR: orange-ish for B
}
BALL_COLOR = (0, 0, 255)
HOOP_COLOR = (0, 255, 0)

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
]


def _pid(s: dict[str, Any]) -> str:
    return str(s.get("stable_player_id") or s.get("track_id"))


def _frame(s: dict[str, Any]) -> int:
    return int(s.get("frame_idx", s.get("frame", 0)))


def _load_pose(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples") or data.get("tracks_pose") or []
    return data


def _load_ball(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames") if isinstance(data, dict) else data
    out = {}
    for fr in frames or []:
        out[int(fr["frame_idx"])] = fr
    return out


def _load_hoop(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wrist(kps: list, idx: int, thresh: float = 0.2) -> list[float] | None:
    if not kps or idx >= len(kps):
        return None
    x, y, c = kps[idx][0], kps[idx][1], kps[idx][2] if len(kps[idx]) > 2 else 0.0
    if c < thresh:
        return None
    return [float(x), float(y)]


def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def compute_relations(
    pose_by_frame: dict[int, list[dict[str, Any]]],
    ball_by_frame: dict[int, dict[str, Any]],
    hoop_center: list[float],
    frame_indices: list[int],
    fps: float,
) -> list[dict[str, Any]]:
    rows = []
    for fidx in frame_indices:
        ball = ball_by_frame.get(fidx)
        ball_c = ball.get("center_xy") if ball else None
        people = { _pid(s): s for s in pose_by_frame.get(fidx, []) }
        rel = {
            "frame_idx": fidx,
            "time_s": round(fidx / fps, 4) if fps > 0 else 0.0,
            "ball_center_xy": ball_c,
            "ball_source": (ball or {}).get("source"),
            "ball_detected": bool((ball or {}).get("detected")),
            "ball_to_hoop_distance_pixel": _dist(ball_c, hoop_center),
            "ball_to_A_left_wrist": None,
            "ball_to_A_right_wrist": None,
            "ball_to_B_left_wrist": None,
            "ball_to_B_right_wrist": None,
        }
        for pid in ("A", "B"):
            s = people.get(pid)
            if not s:
                continue
            kps = s.get("keypoints") or s.get("keypoints_smoothed") or s.get("keypoints_raw") or []
            rel[f"ball_to_{pid}_left_wrist"] = _dist(ball_c, _wrist(kps, LEFT_WRIST))
            rel[f"ball_to_{pid}_right_wrist"] = _dist(ball_c, _wrist(kps, RIGHT_WRIST))
        rows.append(rel)
    return rows


def render_overlay(
    video_path: str,
    pose_samples: list[dict[str, Any]],
    ball_by_frame: dict[int, dict[str, Any]],
    hoop: dict[str, Any],
    output_mp4: Path,
    *,
    trail: int = 15,
) -> dict[str, Any]:
    pose_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in pose_samples:
        pose_by_frame[_frame(s)].append(s)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    hoop_c = hoop.get("hoop_center_pixel")
    hoop_bbox = hoop.get("hoop_bbox_xyxy") or hoop.get("rim_or_backboard_bbox_xyxy")
    ball_trail: deque[tuple[int, int]] = deque(maxlen=trail)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Players
        for s in pose_by_frame.get(frame_idx, []):
            pid = _pid(s)
            color = PLAYER_COLORS.get(pid, (0, 220, 120))
            if s.get("bbox_xyxy"):
                x1, y1, x2, y2 = map(int, s["bbox_xyxy"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, pid, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            kps = s.get("keypoints") or []
            for a, b in SKELETON_EDGES:
                if a < len(kps) and b < len(kps):
                    if len(kps[a]) >= 3 and len(kps[b]) >= 3 and kps[a][2] >= 0.2 and kps[b][2] >= 0.2:
                        cv2.line(
                            frame,
                            (int(kps[a][0]), int(kps[a][1])),
                            (int(kps[b][0]), int(kps[b][1])),
                            color,
                            2,
                        )
            for kp in kps:
                if len(kp) >= 3 and kp[2] >= 0.2:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)

        # Hoop
        if hoop_c is not None:
            hx, hy = int(hoop_c[0]), int(hoop_c[1])
            cv2.drawMarker(frame, (hx, hy), HOOP_COLOR, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
            cv2.circle(frame, (hx, hy), 14, HOOP_COLOR, 2)
            cv2.putText(frame, "HOOP", (hx + 12, hy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HOOP_COLOR, 2)
        if hoop_bbox is not None:
            x1, y1, x2, y2 = map(int, hoop_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), HOOP_COLOR, 1)

        # Ball
        ball = ball_by_frame.get(frame_idx)
        if ball and ball.get("center_xy") is not None:
            cx, cy = int(ball["center_xy"][0]), int(ball["center_xy"][1])
            ball_trail.append((cx, cy))
            pts = list(ball_trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], BALL_COLOR, 2)
            cv2.circle(frame, (cx, cy), 7, BALL_COLOR, -1)
            src = ball.get("source", "?")
            cv2.putText(frame, f"BALL:{src}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1)
            if ball.get("bbox_xyxy"):
                x1, y1, x2, y2 = map(int, ball["bbox_xyxy"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 1)
        else:
            # keep trail decay naturally by not appending
            pass

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {"frames_written": frame_idx, "fps": fps, "size": [w, h], "expected_frames": n}


def main() -> None:
    p = argparse.ArgumentParser(description="Render players+ball+hoop overlay and relation JSON")
    p.add_argument("--video", required=True)
    p.add_argument("--pose-json", default="data/output/1v1_36s/tracks_pose.json")
    p.add_argument("--ball-json", default="data/output/1v1_36s/ball_track.json")
    p.add_argument("--hoop-config", default="configs/hoop_1v1_36s.json")
    p.add_argument("--output-dir", default="data/output/1v1_36s")
    p.add_argument("--trail", type=int, default=15)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose = _load_pose(Path(args.pose_json))
    ball = _load_ball(Path(args.ball_json))
    hoop = _load_hoop(Path(args.hoop_config))
    hoop_c = [float(x) for x in hoop["hoop_center_pixel"]]

    overlay_path = out_dir / "basketball_overlay.mp4"
    info = render_overlay(args.video, pose, ball, hoop, overlay_path, trail=args.trail)

    # Relations over union of pose/ball frame indices (prefer video length from overlay)
    pose_by_frame: dict[int, list] = defaultdict(list)
    for s in pose:
        pose_by_frame[_frame(s)].append(s)
    n = int(info["frames_written"])
    fps = float(info["fps"])
    relations = compute_relations(pose_by_frame, ball, hoop_c, list(range(n)), fps)
    rel_path = out_dir / "ball_player_relations.json"
    rel_path.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "hoop_center_pixel": hoop_c,
                "fps": fps,
                "frames": relations,
                "note": "Pixel-space distances only. Event semantics (possession/shot/make) not labeled yet.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "outputs": {
                    "basketball_overlay": str(overlay_path),
                    "ball_player_relations": str(rel_path),
                },
                "info": info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
