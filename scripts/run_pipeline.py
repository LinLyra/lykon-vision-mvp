#!/usr/bin/env python3
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from lykon.pipeline.baseline import run_baseline
from lykon.court.homography import load_court_config


def main():
    p = argparse.ArgumentParser(description="Run Lykon video-first basketball pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--court-config", default=None, help="Homography court calibration JSON")
    p.add_argument("--court-roi", default=None, help="Court ROI polygon JSON (filters spectators)")
    p.add_argument("--player-init", default=None, help="Human-initialized players JSON")
    p.add_argument("--mode", choices=["motion", "tactical"], default="motion")
    p.add_argument(
        "--config",
        default=None,
        help="YAML config. Defaults to configs/motion_1v1.yaml or configs/tactical_3v3.yaml by mode.",
    )
    args = p.parse_args()

    if args.config:
        config_path = args.config
    else:
        default_cfg = {
            "motion": "configs/motion_1v1.yaml",
            "tactical": "configs/tactical_3v3.yaml",
        }[args.mode]
        config_path = default_cfg if Path(default_cfg).exists() else "configs/default.yaml"
        if not Path(config_path).exists():
            config_path = None

    court = load_court_config(args.court_config) if args.court_config else None
    # Merge ROI file into court config if both provided separately
    court_roi = args.court_roi
    if court_roi is None and court and court.get("court_polygon_pixels"):
        court_roi = court

    summary = run_baseline(
        args.video,
        args.output,
        court_config=court,
        config_path=config_path,
        mode=args.mode,
        player_init=args.player_init,
        court_roi=court_roi,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
