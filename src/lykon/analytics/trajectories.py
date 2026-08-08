from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from lykon.court.homography import compute_homography, image_to_court
from lykon.schema import coerce_sample, foot_point_from_sample, player_id_of


def build_trajectories(samples: list[dict[str, Any]], court_config: dict | None = None) -> list[dict[str, Any]]:
    H = compute_homography(court_config) if court_config else None
    out: list[dict[str, Any]] = []
    for raw in samples:
        s = coerce_sample(raw)
        image_xy = s.get("pixel_foot_point") or foot_point_from_sample(s)
        pid = player_id_of(s)
        item = {
            "frame": int(s["frame_idx"]),
            "frame_idx": int(s["frame_idx"]),
            "time_s": float(s["time_s"]),
            "stable_player_id": pid,
            "track_id": pid,  # legacy key now holds stable id
            "temporary_track_id": s.get("temporary_track_id"),
            "image_xy": [float(image_xy[0]), float(image_xy[1])],
            "tracking_state": s.get("tracking_state"),
        }
        if "court_xy_m" in s:
            item["court_xy_m"] = [float(s["court_xy_m"][0]), float(s["court_xy_m"][1])]
        elif H is not None:
            x, y = image_to_court((image_xy[0], image_xy[1]), H)
            item["court_xy_m"] = [x, y]
        out.append(item)
    return out


def save_trajectories(items: list[dict[str, Any]], output_path: str | Path) -> None:
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[str(item.get("stable_player_id", item.get("track_id")))].append(item)
    payload = {"samples": items, "tracks": grouped}
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
