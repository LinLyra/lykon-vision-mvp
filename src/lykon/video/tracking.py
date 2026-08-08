"""Person detection + short-term MOT + stable Lykon player identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from lykon.court.homography import compute_homography, image_to_court
from lykon.court.roi import draw_roi, filter_detections_by_roi, foot_point_from_bbox
from lykon.pose.recovery import recover_poses
from lykon.schema import make_player_sample, normalize_keypoints
from lykon.tracking.appearance import AppearanceReIDInterface
from lykon.tracking.player_identity import PlayerIdentityManager


TEAM_COLORS = {
    "A": (40, 180, 255),   # orange-ish BGR
    "B": (255, 160, 40),   # blue-ish BGR
    "": (0, 220, 120),
}


def _team_of(player_id: str, team: str | None = None) -> str:
    if team:
        return str(team)
    if player_id and player_id[0] in ("A", "B"):
        return player_id[0]
    return ""


def _load_player_init(path: str | Path | dict | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if isinstance(path, dict):
        return list(path.get("players") or [])
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("players") or data if isinstance(data, list) else [])


def run_pose_tracking(
    video_path: str | Path,
    output_path: str | Path,
    overlay_path: str | Path | None = None,
    pose_model: str = "yolo11m-pose.pt",
    tracker: str = "botsort.yaml",
    conf: float = 0.15,
    iou: float = 0.4,
    imgsz: int = 1280,
    max_players: int = 6,
    max_active_players: int | None = None,
    player_init: str | Path | dict | None = None,
    court_roi: dict | None = None,
    court_config: dict | None = None,
    mode: str = "motion",
    debug_overlay_path: str | Path | None = None,
    recover_pose_gaps: bool = True,
    max_pose_gap: int = 8,
    max_lost_frames: int = 30,
    emit_predicted: bool | None = None,
    # legacy alias
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run detection/tracking, then assign stable Lykon player IDs.

    temporary YOLO track_id is kept separately from stable_player_id.
    max_active_players is applied AFTER court ROI + identity matching,
    not by blindly truncating all detections by bbox area.
    """
    if "max_players" in kwargs and max_active_players is None:
        max_active_players = int(kwargs.get("max_players") or max_players)
    if max_active_players is None:
        max_active_players = max_players

    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(pose_model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    players_init = _load_player_init(player_init)
    if emit_predicted is None:
        # 1v1 motion with human init: keep occluded player visible via Kalman prediction
        emit_predicted = bool(players_init) or mode == "motion"

    identity = PlayerIdentityManager(
        players_init,
        max_lost_frames=max_lost_frames,
        occluded_frames=max(10, min(30, max_lost_frames // 3)),
        mode=mode,
        allow_new_players=False,
        max_players=max_active_players,
        appearance=AppearanceReIDInterface(),
        persistent_roster=bool(players_init) or mode == "motion",
    )

    H = compute_homography(court_config) if court_config else None

    writer = None
    debug_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    if debug_overlay_path is not None:
        debug_overlay_path = Path(debug_overlay_path)
        debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        debug_writer = cv2.VideoWriter(str(debug_overlay_path), fourcc, fps, (width, height))

    samples: list[dict[str, Any]] = []
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        classes=[0],  # person
    )

    for frame_idx, result in enumerate(results):
        frame = result.orig_img.copy()
        time_s = frame_idx / fps
        raw_dets: list[dict[str, Any]] = []

        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                det_conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                temp_id: Optional[int] = None
                if box.id is not None:
                    temp_id = int(box.id.item())

                kps = None
                if keypoints is not None and i < len(keypoints):
                    kp_xy = keypoints.xy[i].cpu().numpy()
                    if keypoints.conf is not None:
                        kp_conf = keypoints.conf[i].cpu().numpy()
                    else:
                        kp_conf = np.ones(len(kp_xy), dtype=float)
                    kps = normalize_keypoints(kp_xy, kp_conf)

                foot = foot_point_from_bbox(xyxy)
                raw_dets.append({
                    "temporary_track_id": temp_id,
                    "bbox_xyxy": [float(x) for x in xyxy],
                    "detection_confidence": det_conf,
                    "keypoints": kps,
                    "pixel_foot_point": foot,
                })

        active, ignored = filter_detections_by_roi(raw_dets, court_roi)
        matched = identity.update(active, frame_bgr=frame)

        # Cap to max_active_players by preferring initialized IDs / higher confidence
        if len(matched) > max_active_players:
            init_ids = {str(p["player_id"]) for p in players_init}
            matched.sort(
                key=lambda d: (
                    0 if d.get("stable_player_id") in init_ids else 1,
                    -float(d.get("detection_confidence") or 0.0),
                )
            )
            matched = matched[:max_active_players]

        if emit_predicted:
            matched_ids = {str(d.get("stable_player_id")) for d in matched}
            for pred in identity.predicted_missing():
                if str(pred.get("stable_player_id")) not in matched_ids:
                    matched.append(pred)

        frame_vis = draw_roi(frame, court_roi)
        frame_dbg = frame_vis.copy()

        # Draw ignored persons (debug)
        for det in ignored:
            x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
            cv2.rectangle(frame_dbg, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(frame_dbg, "IGNORED", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        for det in matched:
            pid = str(det.get("stable_player_id", "?"))
            team = _team_of(pid, det.get("team"))
            color = TEAM_COLORS.get(team, TEAM_COLORS[""])
            bbox = det["bbox_xyxy"]
            x1, y1, x2, y2 = map(int, bbox)
            kps = det.get("keypoints")
            pose_source = "missing" if det.get("predicted") or kps is None else "detected"
            tracking_state = det.get("tracking_state", "tracked")

            court_xy = None
            foot = det.get("pixel_foot_point") or foot_point_from_bbox(bbox)
            if H is not None:
                court_xy = list(image_to_court((foot[0], foot[1]), H))

            sample = make_player_sample(
                frame_idx=frame_idx,
                time_s=time_s,
                stable_player_id=pid,
                bbox_xyxy=bbox,
                detection_confidence=float(det.get("detection_confidence") or 0.0),
                keypoints=kps,
                temporary_track_id=det.get("temporary_track_id"),
                pose_source=pose_source,  # type: ignore[arg-type]
                tracking_state=tracking_state,  # type: ignore[arg-type]
                court_xy_m=court_xy,
                pixel_foot_point=foot,
                team=team,
            )
            samples.append(sample)

            # Main overlay: stable ID (dashed-ish thinner box when predicted)
            thickness = 1 if det.get("predicted") else 2
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
            label_main = f"{pid}*" if det.get("predicted") else pid
            cv2.putText(frame_vis, label_main, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if sample["pose_valid"] and sample["keypoints"] and not det.get("predicted"):
                for x, y, c in sample["keypoints"]:
                    if c >= 0.2 and x > 0 and y > 0:
                        cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 180, 255), -1)

            # Debug overlay
            cv2.rectangle(frame_dbg, (x1, y1), (x2, y2), color, 2)
            temp = det.get("temporary_track_id")
            label = f"{pid}  tmp={temp}  {tracking_state}  pose={sample['pose_source']}"
            cv2.putText(frame_dbg, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(
                frame_dbg,
                f"conf={sample['detection_confidence']:.2f}",
                (x1, min(height - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        if writer is not None:
            writer.write(frame_vis)
        if debug_writer is not None:
            debug_writer.write(frame_dbg)

    cap.release()
    if writer is not None:
        writer.release()
    if debug_writer is not None:
        debug_writer.release()

    if recover_pose_gaps:
        samples = recover_poses(samples, max_gap=max_pose_gap)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return samples
