#!/usr/bin/env python3
"""1v1 motion tracking-only runner (no court / tactics / WHAM / grid search).

Reads an existing player-init JSON, runs fixed-config YOLO pose tracking,
applies light pose smoothing, writes overlays + tracks_pose + metrics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import yaml

from lykon.pose.smoothing import apply_pose_smoothing
from lykon.video.io import probe_video
from lykon.video.tracking import run_pose_tracking

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "evaluate_tracking",
    Path(__file__).resolve().parent / "evaluate_tracking.py",
)
_ev = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ev)


def main() -> None:
    p = argparse.ArgumentParser(description="Run 1v1 motion tracking + pose (fixed config)")
    p.add_argument("--video", required=True)
    p.add_argument("--player-init", required=True, help="configs/players_1v1_36s.json")
    p.add_argument("--config", default="configs/motion_1v1_36s.yaml")
    p.add_argument("--output", default="data/output/1v1_36s")
    p.add_argument("--smooth-method", default=None, choices=["savgol", "one_euro", "none"])
    args = p.parse_args()

    if not Path(args.player_init).exists():
        raise SystemExit(f"Missing player init: {args.player_init}\nRun scripts/init_players.py first.")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    v = dict(cfg.get("video") or {})
    sm = dict(cfg.get("pose_smoothing") or {})
    method = args.smooth_method or sm.get("method", "savgol")
    window = int(sm.get("window", 5))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = probe_video(args.video)

    samples = run_pose_tracking(
        args.video,
        out / "tracks_pose_raw.json",
        overlay_path=out / "tracked_overlay.mp4",
        debug_overlay_path=out / "tracking_debug.mp4",
        pose_model=v.get("pose_model", "yolo11m-pose.pt"),
        tracker=v.get("tracker", "botsort.yaml"),
        conf=float(v.get("conf", 0.20)),
        iou=float(v.get("iou", 0.40)),
        imgsz=int(v.get("imgsz", 960)),
        max_players=int(v.get("max_players", 2)),
        max_active_players=int(v.get("max_active_players", 2)),
        max_lost_frames=int(v.get("max_lost_frames", 120)),
        max_pose_gap=int(v.get("max_pose_gap", 12)),
        recover_pose_gaps=bool(v.get("recover_pose_gaps", True)),
        emit_predicted=bool(v.get("emit_predicted", True)),
        player_init=args.player_init,
        mode="motion",
    )

    samples = apply_pose_smoothing(samples, method=method, window=window)  # type: ignore[arg-type]
    tracks_path = out / "tracks_pose.json"
    tracks_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = _ev.evaluate_tracks(samples, total_frames=info.frame_count)
    metrics["duration_s"] = round(float(info.duration_s), 3)
    metrics["video"] = {
        "path": str(args.video),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
    }
    metrics["params"] = {
        "pose_model": v.get("pose_model", "yolo11m-pose.pt"),
        "imgsz": int(v.get("imgsz", 960)),
        "conf": float(v.get("conf", 0.20)),
        "iou": float(v.get("iou", 0.40)),
        "tracker": v.get("tracker", "botsort.yaml"),
        "max_active_players": 2,
        "max_lost_frames": int(v.get("max_lost_frames", 30)),
        "max_pose_gap": int(v.get("max_pose_gap", 8)),
        "smooth_method": method,
        "smooth_window": window,
        "player_init": args.player_init,
    }
    metrics["outputs"] = {
        "tracks_pose": str(tracks_path),
        "tracked_overlay": str(out / "tracked_overlay.mp4"),
        "tracking_debug": str(out / "tracking_debug.mp4"),
    }
    (out / "tracking_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
