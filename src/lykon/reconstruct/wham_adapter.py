from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


def run_wham(video_path: str, wham_dir: str, output_dir: str, local_only: bool = False, run_smplify: bool = False) -> dict[str, Any]:
    wham = Path(wham_dir).resolve()
    video = Path(video_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    demo = wham / "demo.py"
    if not demo.exists():
        raise FileNotFoundError(f"WHAM demo.py not found at {demo}")

    cmd = ["python", "demo.py", "--video", str(video), "--visualize"]
    if local_only:
        cmd.append("--estimate_local_only")
    if run_smplify:
        cmd.append("--run_smplify")

    proc = subprocess.run(cmd, cwd=str(wham), capture_output=True, text=True)
    log = out / "wham.log"
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")

    # WHAM output layout can vary by release/config. We discover newly relevant
    # artifacts instead of assuming one hard-coded filename.
    candidates = []
    for ext in ("*.mp4", "*.pkl", "*.npz", "*.json"):
        candidates.extend(wham.rglob(ext))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:30]

    copied = []
    for p in candidates:
        if p.resolve() == video:
            continue
        try:
            target = out / p.name
            if p.is_file() and p.stat().st_size > 0:
                shutil.copy2(p, target)
                copied.append(str(target))
                if len(copied) >= 8:
                    break
        except Exception:
            pass

    summary = {
        "returncode": proc.returncode,
        "command": cmd,
        "log": str(log),
        "copied_artifacts": copied,
        "note": "WHAM requires its official checkpoints and licensed SMPL/SMPLify assets. Check wham.log if setup is incomplete.",
    }
    (out / "wham_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"WHAM failed. See {log}")
    return summary
