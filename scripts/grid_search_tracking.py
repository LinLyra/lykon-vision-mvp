#!/usr/bin/env python3
"""Parameter grid search for motion vs tactical tracking configs.

Does not add product features — only sweeps existing tracking knobs and
scores with evaluate_tracking metrics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import yaml

from lykon.video.io import probe_video
from lykon.video.tracking import run_pose_tracking

# Import evaluator without going through script CLI
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "evaluate_tracking",
    Path(__file__).resolve().parent / "evaluate_tracking.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
evaluate_tracks = _mod.evaluate_tracks


MODELS = ["yolo11m-pose.pt", "yolo11l-pose.pt"]
IMGSZ = [960, 1280, 1536]
CONFS = [0.10, 0.15, 0.20, 0.25]
TRACKERS = ["botsort.yaml", "bytetrack.yaml"]


def score_report(report: dict[str, Any], mode: str, expected_players: int) -> float:
    """Higher is better. Scene-specific priorities."""
    n_players = len(report.get("players") or [])
    player_gap = abs(n_players - expected_players)

    if mode == "motion":
        # 1v1: near-zero stable ID switches is king; keep ~2 players
        return (
            -100.0 * report["stable_id_switches"]
            - 15.0 * report["temporary_id_switches"]
            + 40.0 * report["player_detection_coverage"]
            + 35.0 * report["pose_coverage"]
            + 20.0 * report["stable_id_coverage"]
            - 25.0 * player_gap
            - 0.05 * sum((report.get("lost_frames_per_player") or {}).values())
            + 5.0 * report["average_detection_confidence"]
        )

    # tactical 3v3: coverage of multiple players + low stable switches
    return (
        -80.0 * report["stable_id_switches"]
        - 8.0 * report["temporary_id_switches"]
        + 55.0 * report["player_detection_coverage"]
        + 25.0 * report["pose_coverage"]
        + 15.0 * report["stable_id_coverage"]
        - 12.0 * player_gap
        - 0.03 * sum((report.get("lost_frames_per_player") or {}).values())
        + 3.0 * report["average_detection_confidence"]
        + 2.0 * min(n_players, expected_players)  # reward finding more real players
    )


def run_one(
    video: str,
    out_dir: Path,
    *,
    pose_model: str,
    imgsz: int,
    conf: float,
    tracker: str,
    mode: str,
    max_players: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / "tracks_pose.json"
    t0 = time.time()
    samples = run_pose_tracking(
        video,
        tracks_path,
        overlay_path=None,
        pose_model=pose_model,
        tracker=tracker,
        conf=conf,
        iou=0.4,
        imgsz=imgsz,
        max_players=max_players,
        max_active_players=max_players,
        mode=mode,
        recover_pose_gaps=False,  # grid: score raw detect/identity; recovery is post-process
        debug_overlay_path=None,
    )
    elapsed = time.time() - t0
    info = probe_video(video)
    report = evaluate_tracks(samples, total_frames=info.frame_count)
    report["elapsed_s"] = round(elapsed, 2)
    report["params"] = {
        "pose_model": pose_model,
        "imgsz": imgsz,
        "conf": conf,
        "tracker": tracker.replace(".yaml", ""),
        "mode": mode,
        "max_active_players": max_players,
    }
    report["score"] = round(score_report(report, mode, max_players), 4)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_scene_yaml(path: Path, mode: str, best: dict[str, Any]) -> None:
    p = best["params"]
    payload = {
        "mode": mode,
        "video": {
            "pose_model": p["pose_model"],
            "tracker": f"{p['tracker']}.yaml" if not str(p["tracker"]).endswith(".yaml") else p["tracker"],
            "conf": float(p["conf"]),
            "iou": 0.4,
            "imgsz": int(p["imgsz"]),
            "max_active_players": 2 if mode == "motion" else 6,
            "max_players": 2 if mode == "motion" else 6,
            "max_lost_frames": 30 if mode == "motion" else 25,
            "max_pose_gap": 8,
            "recover_pose_gaps": True,
        },
        "analytics": {
            "smoothing_window": 5,
            "shot_min_arm_raise_px": 30,
            "shot_cooldown_s": 1.0,
        },
        "render": {"fps": 30, "pseudo3d_depth_scale": 0.35},
        "notes": (
            f"Selected by grid search. score={best['score']} "
            f"stable_id_switches={best['stable_id_switches']} "
            f"pose_coverage={best['pose_coverage']} "
            f"detection_coverage={best['player_detection_coverage']}"
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", choices=["motion", "tactical"], required=True)
    ap.add_argument("--output", required=True, help="Directory for grid results")
    ap.add_argument("--config-out", default=None, help="Write best YAML here")
    ap.add_argument("--max-players", type=int, default=None)
    args = ap.parse_args()

    max_players = args.max_players or (2 if args.mode == "motion" else 6)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(MODELS, IMGSZ, CONFS, TRACKERS))
    results: list[dict[str, Any]] = []
    print(f"Grid: {len(combos)} runs | mode={args.mode} | video={args.video}")

    for i, (model, imgsz, conf, tracker) in enumerate(combos, 1):
        tag = f"{Path(model).stem}_s{imgsz}_c{conf}_{Path(tracker).stem}"
        run_dir = root / tag
        print(f"[{i}/{len(combos)}] {tag}", flush=True)
        try:
            report = run_one(
                args.video,
                run_dir,
                pose_model=model,
                imgsz=imgsz,
                conf=conf,
                tracker=tracker,
                mode=args.mode,
                max_players=max_players,
            )
            print(
                f"  score={report['score']} "
                f"stable_sw={report['stable_id_switches']} "
                f"det={report['player_detection_coverage']} "
                f"pose={report['pose_coverage']} "
                f"players={report.get('players')} "
                f"t={report['elapsed_s']}s",
                flush=True,
            )
            results.append(report)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({
                "params": {
                    "pose_model": model,
                    "imgsz": imgsz,
                    "conf": conf,
                    "tracker": tracker.replace(".yaml", ""),
                    "mode": args.mode,
                },
                "error": str(e),
                "score": -1e9,
            })

    results_sorted = sorted(results, key=lambda r: r.get("score", -1e9), reverse=True)
    summary = {
        "video": args.video,
        "mode": args.mode,
        "n_runs": len(results_sorted),
        "best": results_sorted[0] if results_sorted else None,
        "top5": results_sorted[:5],
        "all": results_sorted,
    }
    (root / "grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.config_out and results_sorted and "error" not in results_sorted[0]:
        write_scene_yaml(Path(args.config_out), args.mode, results_sorted[0])
        print(f"Wrote best config -> {args.config_out}")

    print(json.dumps({"best": summary["best"], "top5_scores": [x.get("score") for x in summary["top5"]]}, indent=2))


if __name__ == "__main__":
    main()
