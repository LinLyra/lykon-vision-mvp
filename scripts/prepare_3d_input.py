#!/usr/bin/env python3
"""Prepare / validate 3D reconstruction inputs from 1v1 motion tracking outputs.

Does NOT run WHAM / SMPL. Only checks crops + pose integrity and writes a manifest.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


def _pid(s: dict[str, Any]) -> str:
    return str(s.get("stable_player_id") or s.get("track_id"))


def _frame(s: dict[str, Any]) -> int:
    return int(s.get("frame_idx", s.get("frame", 0)))


def _probe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "readable": False}
    info = {
        "path": str(path),
        "readable": True,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def _longest_missing_gap(rows: list[dict[str, Any]]) -> int:
    """Longest consecutive run of pose_source == missing (or invalid detected)."""
    if not rows:
        return 0
    rows = sorted(rows, key=_frame)
    frames = {_frame(r): r for r in rows}
    f0, f1 = min(frames), max(frames)
    longest = 0
    cur = 0
    for f in range(f0, f1 + 1):
        r = frames.get(f)
        missing = (
            r is None
            or r.get("pose_source") == "missing"
            or (not r.get("pose_valid", True) and r.get("pose_source") != "interpolated")
        )
        if missing:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def _pose_coverage(rows: list[dict[str, Any]], total_frames: int) -> float:
    if total_frames <= 0:
        return 0.0
    ok = sum(
        1
        for r in rows
        if r.get("pose_valid") or r.get("pose_source") in ("detected", "interpolated")
    )
    # coverage vs player present frames is more honest; also report vs timeline
    present = len({_frame(r) for r in rows})
    return round(ok / max(present, 1), 4)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate 1v1 motion outputs for later WHAM/SMPL")
    p.add_argument("--video", required=True, help="Original full-frame video")
    p.add_argument("--pose-json", required=True)
    p.add_argument("--output-dir", default="data/output/1v1_36s")
    p.add_argument("--players", default="A,B")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    samples = json.loads(Path(args.pose_json).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("tracks_pose") or []

    by_player: dict[str, list] = defaultdict(list)
    for s in samples:
        by_player[_pid(s)].append(s)

    src = _probe(Path(args.video)) or {}
    fps = float(src.get("fps") or 30.0)
    total_frames = int(src.get("frame_count") or 0)

    # stable id integrity: only A/B expected
    players = [x.strip() for x in args.players.split(",") if x.strip()]
    unexpected = sorted(set(by_player) - set(players))
    stable_ok = len(unexpected) == 0 and all(pid in by_player for pid in players)

    player_manifest: dict[str, Any] = {}
    for pid in players:
        rows = by_player.get(pid, [])
        crop_path = out_dir / f"player_{pid}.mp4"
        crop = _probe(crop_path)
        gap = _longest_missing_gap(rows)
        pose_cov = _pose_coverage(rows, total_frames)
        ready = bool(
            crop
            and crop.get("readable")
            and crop.get("frame_count", 0) > 0
            and pose_cov >= 0.5
            and gap <= 30
            and pid in by_player
        )
        player_manifest[pid] = {
            "crop_video": str(crop_path),
            "crop_info": crop,
            "pose_json": str(args.pose_json),
            "samples": len(rows),
            "pose_coverage": pose_cov,
            "longest_missing_pose_gap": gap,
            "ready_for_3d": ready,
        }

    manifest = {
        "video": str(args.video),
        "fps": fps,
        "frame_count": total_frames,
        "source_video": src,
        "stable_id_integrity": stable_ok,
        "unexpected_players": unexpected,
        "players": player_manifest,
        "note": "Preparation only. WHAM/SMPL not executed.",
    }
    out_path = out_dir / "3d_input_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
