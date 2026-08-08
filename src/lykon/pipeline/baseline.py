from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml

from lykon.video.io import probe_video
from lykon.video.tracking import run_pose_tracking
from lykon.analytics.trajectories import build_trajectories, save_trajectories
from lykon.analytics.metrics import compute_motion_metrics
from lykon.analytics.shots import detect_shot_motion_candidates
from lykon.analytics.tactics import summarize_1v1_tactics
from lykon.reconstruct.pseudo3d import build_pseudo3d, save_pseudo3d
from lykon.render.pseudo3d_video import render_pseudo3d_video
from lykon.render.overlay import render_court_topdown
from lykon.court.roi import load_court_roi


MODE_DEFAULTS = {
    "motion": {
        "video": {
            "pose_model": "yolo11m-pose.pt",
            "tracker": "botsort.yaml",
            "conf": 0.25,
            "iou": 0.4,
            "imgsz": 1280,
            "max_active_players": 2,
            "max_players": 2,
            "max_lost_frames": 30,
            "max_pose_gap": 8,
        },
    },
    "tactical": {
        "video": {
            "pose_model": "yolo11m-pose.pt",
            "tracker": "botsort.yaml",
            "conf": 0.15,
            "iou": 0.4,
            "imgsz": 1280,
            "max_active_players": 6,
            "max_players": 6,
            "max_lost_frames": 25,
            "max_pose_gap": 8,
        },
    },
}


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(config_path: str | None = None, mode: str = "motion") -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "mode": mode,
        "video": {
            "pose_model": "yolo11m-pose.pt",
            "tracker": "botsort.yaml",
            "conf": 0.15,
            "iou": 0.4,
            "imgsz": 1280,
            "max_active_players": 6 if mode == "tactical" else 2,
            "max_players": 6 if mode == "tactical" else 2,
            "max_lost_frames": 30,
            "max_pose_gap": 8,
            "recover_pose_gaps": True,
        },
        "analytics": {"smoothing_window": 5, "shot_cooldown_s": 1.0},
        "render": {"fps": 30, "pseudo3d_depth_scale": 0.35},
    }
    cfg = _deep_update(cfg, MODE_DEFAULTS.get(mode, {}))
    if config_path and Path(config_path).exists():
        user_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        cfg = _deep_update(cfg, user_cfg)
    cfg["mode"] = mode
    return cfg


def run_baseline(
    video_path: str,
    output_dir: str,
    court_config: dict | None = None,
    config_path: str | None = None,
    *,
    mode: str = "motion",
    player_init: str | None = None,
    court_roi: str | dict | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_pipeline_config(config_path, mode=mode)

    roi = load_court_roi(court_roi) if court_roi else None
    # Allow court_config to also carry ROI polygon
    if roi is None and court_config and court_config.get("court_polygon_pixels"):
        roi = court_config

    info = probe_video(video_path)
    pose_json = root / "tracks_pose.json"
    overlay_mp4 = root / "tracked_overlay.mp4"
    debug_mp4 = root / "tracking_debug.mp4"

    video_kwargs = dict(cfg["video"])
    # Strip non-tracking keys that may appear in yaml
    for k in ("mode",):
        video_kwargs.pop(k, None)

    samples = run_pose_tracking(
        video_path,
        pose_json,
        overlay_mp4,
        player_init=player_init,
        court_roi=roi,
        court_config=court_config,
        mode=mode,
        debug_overlay_path=debug_mp4,
        **video_kwargs,
    )

    trajectories = build_trajectories(samples, court_config=court_config)
    save_trajectories(trajectories, root / "trajectories.json")

    metrics = compute_motion_metrics(trajectories, smoothing_window=cfg["analytics"]["smoothing_window"])
    tactics = summarize_1v1_tactics(trajectories)
    metrics["tactics_geometry"] = tactics
    (root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    events = detect_shot_motion_candidates(samples, cooldown_s=cfg["analytics"]["shot_cooldown_s"])
    (root / "events.json").write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")

    pseudo = build_pseudo3d(samples, trajectories, depth_scale=cfg["render"]["pseudo3d_depth_scale"])
    pseudo_json = root / "pseudo3d.json"
    save_pseudo3d(pseudo, pseudo_json)
    replay = root / "pseudo3d_replay.mp4"
    render_pseudo3d_video(pseudo_json, replay, fps=min(info.fps, cfg["render"]["fps"]))

    topdown = root / "court_topdown.mp4"
    topdown_path = render_court_topdown(
        samples,
        video_path,
        topdown,
        court_config=court_config,
    )

    players_seen = sorted({str(x.get("stable_player_id", x.get("track_id"))) for x in samples})
    summary = {
        "video": info.__dict__,
        "mode": mode,
        "players_seen": players_seen,
        "pose_samples": len(samples),
        "shot_motion_candidates": len(events),
        "court_calibrated": court_config is not None,
        "court_roi": roi is not None,
        "player_init": player_init,
        "outputs": {
            "tracked_overlay": str(overlay_mp4),
            "tracking_debug": str(debug_mp4),
            "court_topdown": str(topdown_path) if topdown_path else None,
            "trajectories": str(root / "trajectories.json"),
            "metrics": str(root / "metrics.json"),
            "events": str(root / "events.json"),
            "tracks_pose": str(pose_json),
            "pseudo3d_replay": str(replay),
        },
        "disclaimer": "Pseudo-3D and shot candidates are prototype outputs. Use WHAM/SMPL and ball semantics for higher-fidelity reconstruction and confirmed basketball events.",
    }
    (root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
