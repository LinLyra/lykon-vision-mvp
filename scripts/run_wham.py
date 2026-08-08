#!/usr/bin/env python3
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from lykon.reconstruct.wham_adapter import run_wham


def main():
    p = argparse.ArgumentParser(description="Run official WHAM as Lykon's high-fidelity 3D backend")
    p.add_argument("--video", required=True)
    p.add_argument("--wham-dir", default="external/WHAM")
    p.add_argument("--output", required=True)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--run-smplify", action="store_true")
    args = p.parse_args()
    result = run_wham(args.video, args.wham_dir, args.output, args.local_only, args.run_smplify)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
