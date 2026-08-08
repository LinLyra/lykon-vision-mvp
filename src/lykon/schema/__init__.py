"""Unified tracking / pose sample schema for Lykon Vision."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

PoseSource = Literal["detected", "interpolated", "missing"]
TrackingState = Literal["tracked", "occluded", "lost", "recovered"]

NUM_KEYPOINTS = 17


def normalize_keypoints(raw: Any, conf: Any = None) -> list[list[float]]:
    """Force every keypoint to [x, y, confidence].

    Accepts:
    - list of [x, y]
    - list of [x, y, conf]
    - keypoints + separate keypoint_conf arrays
    """
    if raw is None:
        return [[0.0, 0.0, 0.0] for _ in range(NUM_KEYPOINTS)]

    kps = list(raw)
    confs: list[float] | None = None
    if conf is not None:
        confs = [float(c) for c in list(conf)]

    out: list[list[float]] = []
    for i, kp in enumerate(kps):
        vals = list(kp)
        if len(vals) >= 3:
            x, y, c = float(vals[0]), float(vals[1]), float(vals[2])
        elif len(vals) == 2:
            x, y = float(vals[0]), float(vals[1])
            c = float(confs[i]) if confs is not None and i < len(confs) else (1.0 if x > 0 or y > 0 else 0.0)
        else:
            x, y, c = 0.0, 0.0, 0.0
        out.append([x, y, c])

    while len(out) < NUM_KEYPOINTS:
        out.append([0.0, 0.0, 0.0])
    return out[:NUM_KEYPOINTS]


def foot_point_from_bbox(bbox_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [(x1 + x2) / 2.0, y2]


def foot_point_from_sample(sample: dict[str, Any], conf_thresh: float = 0.25) -> list[float]:
    """Prefer ankle keypoints; fall back to bbox bottom-center."""
    kps = sample.get("keypoints") or []
    candidates = []
    for idx in (15, 16):  # left / right ankle (COCO)
        if idx < len(kps) and len(kps[idx]) >= 3 and kps[idx][2] >= conf_thresh:
            candidates.append((kps[idx][0], kps[idx][1]))
    if candidates:
        xs = [p[0] for p in candidates]
        ys = [p[1] for p in candidates]
        return [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))]
    bbox = sample.get("bbox_xyxy") or sample.get("bbox")
    if bbox is None:
        return [0.0, 0.0]
    return foot_point_from_bbox(bbox)


def pose_valid_from_keypoints(keypoints: list[list[float]], min_visible: int = 4, conf_thresh: float = 0.2) -> bool:
    visible = sum(1 for kp in keypoints if len(kp) >= 3 and kp[2] >= conf_thresh)
    return visible >= min_visible


def make_player_sample(
    *,
    frame_idx: int,
    time_s: float,
    stable_player_id: str,
    bbox_xyxy: list[float],
    detection_confidence: float = 0.0,
    keypoints: Any = None,
    keypoint_conf: Any = None,
    temporary_track_id: Optional[int] = None,
    pose_source: PoseSource = "detected",
    tracking_state: TrackingState = "tracked",
    court_xy_m: Optional[list[float]] = None,
    pixel_foot_point: Optional[list[float]] = None,
    team: Optional[str] = None,
) -> dict[str, Any]:
    """Build one unified per-frame player record."""
    kps = normalize_keypoints(keypoints, keypoint_conf)
    if pose_source == "missing":
        pose_valid = False
    elif pose_source == "interpolated":
        pose_valid = pose_valid_from_keypoints(kps)
    else:
        pose_valid = pose_valid_from_keypoints(kps)

    bbox = [float(x) for x in bbox_xyxy]
    foot = pixel_foot_point if pixel_foot_point is not None else foot_point_from_bbox(bbox)
    # Prefer ankle-based foot when pose is usable
    if pose_valid:
        tmp = {"keypoints": kps, "bbox_xyxy": bbox}
        foot = foot_point_from_sample(tmp)

    sample: dict[str, Any] = {
        "frame_idx": int(frame_idx),
        "time_s": float(time_s),
        # Backward-compatible aliases used by older modules / docs
        "frame": int(frame_idx),
        "timestamp_s": float(time_s),
        "temporary_track_id": temporary_track_id,
        "stable_player_id": str(stable_player_id),
        # Legacy int-ish id: prefer hashed stable id for grouping when needed
        "track_id": str(stable_player_id),
        "bbox_xyxy": bbox,
        "bbox": bbox,
        "detection_confidence": float(detection_confidence),
        "confidence": float(detection_confidence),
        "keypoints": kps,
        "pose_valid": bool(pose_valid),
        "pose_source": pose_source,
        "pixel_foot_point": [float(foot[0]), float(foot[1])],
        "tracking_state": tracking_state,
    }
    if court_xy_m is not None:
        sample["court_xy_m"] = [float(court_xy_m[0]), float(court_xy_m[1])]
    if team is not None:
        sample["team"] = team
    return sample


def player_id_of(sample: dict[str, Any]) -> str:
    """Resolve stable player id with legacy fallbacks."""
    if sample.get("stable_player_id") is not None:
        return str(sample["stable_player_id"])
    if sample.get("track_id") is not None:
        return str(sample["track_id"])
    if sample.get("temporary_track_id") is not None:
        return str(sample["temporary_track_id"])
    raise KeyError("sample has no player id field")


def frame_of(sample: dict[str, Any]) -> int:
    if "frame_idx" in sample:
        return int(sample["frame_idx"])
    return int(sample["frame"])


def time_of(sample: dict[str, Any]) -> float:
    if "time_s" in sample:
        return float(sample["time_s"])
    if "timestamp_s" in sample:
        return float(sample["timestamp_s"])
    return 0.0


def coerce_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize a possibly-legacy sample into the unified schema."""
    kps = normalize_keypoints(sample.get("keypoints"), sample.get("keypoint_conf"))
    bbox = sample.get("bbox_xyxy") or sample.get("bbox") or [0, 0, 0, 0]
    sid = player_id_of(sample) if (
        "stable_player_id" in sample or "track_id" in sample or "temporary_track_id" in sample
    ) else "UNKNOWN"
    frame_idx = frame_of(sample) if ("frame_idx" in sample or "frame" in sample) else 0
    time_s = time_of(sample)
    pose_source = sample.get("pose_source", "detected")
    return make_player_sample(
        frame_idx=frame_idx,
        time_s=time_s,
        stable_player_id=sid,
        bbox_xyxy=list(bbox),
        detection_confidence=float(sample.get("detection_confidence", sample.get("confidence", 0.0))),
        keypoints=kps,
        temporary_track_id=sample.get("temporary_track_id"),
        pose_source=pose_source,  # type: ignore[arg-type]
        tracking_state=sample.get("tracking_state", "tracked"),  # type: ignore[arg-type]
        court_xy_m=sample.get("court_xy_m"),
        pixel_foot_point=sample.get("pixel_foot_point"),
        team=sample.get("team"),
    )
