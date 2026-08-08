#!/usr/bin/env python3
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from lykon.court.calibrate import calibrate_from_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width-m", type=float, default=15.0)
    p.add_argument("--depth-m", type=float, default=14.0)
    args = p.parse_args()
    config = calibrate_from_video(args.video, args.output, args.width_m, args.depth_m)
    print(config)


if __name__ == "__main__":
    main()
