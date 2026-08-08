#!/usr/bin/env python3
"""Evaluate tracking / identity / pose coverage metrics."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def evaluate_tracks(samples: list[dict[str, Any]], total_frames: int | None = None) -> dict[str, Any]:
    if not samples:
        return {
            "total_frames": total_frames or 0,
            "player_detection_coverage": 0.0,
            "pose_coverage": 0.0,
            "stable_id_coverage": 0.0,
            "temporary_id_switches": 0,
            "stable_id_switches": 0,
            "lost_frames_per_player": {},
            "interpolated_frames": 0,
            "average_detection_confidence": 0.0,
        }

    frames = sorted({int(s.get("frame_idx", s.get("frame", 0))) for s in samples})
    if total_frames is None:
        total_frames = (max(frames) - min(frames) + 1) if frames else 0

    by_frame: dict[int, list[dict]] = defaultdict(list)
    by_player: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        f = int(s.get("frame_idx", s.get("frame", 0)))
        pid = str(s.get("stable_player_id", s.get("track_id")))
        by_frame[f].append(s)
        by_player[pid].append(s)

    frames_with_players = sum(1 for f in range(min(frames), max(frames) + 1) if by_frame.get(f))
    player_detection_coverage = frames_with_players / max(total_frames, 1)

    pose_ok = sum(1 for s in samples if s.get("pose_valid") or s.get("pose_source") == "detected")
    pose_coverage = pose_ok / max(len(samples), 1)

    stable_ok = sum(1 for s in samples if s.get("stable_player_id") or s.get("track_id") is not None)
    stable_id_coverage = stable_ok / max(len(samples), 1)

    interpolated = sum(1 for s in samples if s.get("pose_source") == "interpolated")
    confs = [float(s.get("detection_confidence", s.get("confidence", 0.0)) or 0.0) for s in samples]
    avg_conf = sum(confs) / max(len(confs), 1)

    # temporary ID switches: same stable id changes temporary track id
    temp_switches = 0
    for pid, rows in by_player.items():
        rows = sorted(rows, key=lambda r: int(r.get("frame_idx", r.get("frame", 0))))
        last = None
        for r in rows:
            tid = r.get("temporary_track_id")
            if tid is None:
                continue
            if last is not None and tid != last:
                temp_switches += 1
            last = tid

    # stable ID switches: same temporary track mapped to different stable ids (identity flicker)
    stable_switches = 0
    by_temp: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        tid = s.get("temporary_track_id")
        if tid is None:
            continue
        by_temp[int(tid)].append(s)
    for tid, rows in by_temp.items():
        rows = sorted(rows, key=lambda r: int(r.get("frame_idx", r.get("frame", 0))))
        last = None
        for r in rows:
            pid = str(r.get("stable_player_id", r.get("track_id")))
            if last is not None and pid != last:
                stable_switches += 1
            last = pid

    # Also count identity flips between consecutive frames for same spatial continuity is hard;
    # report per-player lost/occluded frames
    lost_frames = {}
    for pid, rows in by_player.items():
        lost = sum(1 for r in rows if r.get("tracking_state") in ("lost", "occluded"))
        lost_frames[pid] = lost

    # Per-player detection / pose coverage over the full timeline span
    f0, f1 = min(frames), max(frames)
    span = max(total_frames, f1 - f0 + 1)
    per_player: dict[str, Any] = {}
    for pid, rows in by_player.items():
        # count presence (any row) as detection coverage for that player
        present = {int(r.get("frame_idx", r.get("frame", 0))) for r in rows}
        pose_frames = {
            int(r.get("frame_idx", r.get("frame", 0)))
            for r in rows
            if r.get("pose_valid") or r.get("pose_source") in ("detected", "interpolated")
        }
        per_player[pid] = {
            "detection_coverage": round(len(present) / max(span, 1), 4),
            "pose_coverage": round(len(pose_frames) / max(span, 1), 4),
            "lost_frames": lost_frames.get(pid, 0),
            "samples": len(rows),
        }

    report = {
        "total_frames": total_frames,
        "duration_s": round(
            (
                max(float(s.get("time_s", s.get("timestamp_s", 0.0))) for s in samples)
                - min(float(s.get("time_s", s.get("timestamp_s", 0.0))) for s in samples)
            ),
            3,
        ) if samples else 0.0,
        "players": sorted(by_player.keys()),
        "player_detection_coverage": round(player_detection_coverage, 4),
        "pose_coverage": round(pose_coverage, 4),
        "stable_id_coverage": round(stable_id_coverage, 4),
        "temporary_id_switches": temp_switches,
        "stable_id_switches": stable_switches,
        "lost_frames_per_player": lost_frames,
        "interpolated_frames": interpolated,
        "average_detection_confidence": round(avg_conf, 4),
        "samples": len(samples),
        "per_player": per_player,
    }
    # Convenience aliases requested for 1v1 motion
    for pid in ("A", "B"):
        if pid in per_player:
            report[f"{pid}_detection_coverage"] = per_player[pid]["detection_coverage"]
            report[f"{pid}_pose_coverage"] = per_player[pid]["pose_coverage"]
            report[f"lost_frames_{pid}"] = per_player[pid]["lost_frames"]
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True, help="tracks_pose.json from a pipeline run")
    p.add_argument("--total-frames", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    samples = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("tracks_pose") or []
    report = evaluate_tracks(samples, total_frames=args.total_frames)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
