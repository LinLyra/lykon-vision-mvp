"""Stable Lykon player identity — temporary track_id ≠ permanent player_id."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from lykon.tracking.appearance import AppearanceReIDInterface, color_histogram_distance


class TrackState(str, Enum):
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    LOST = "lost"
    RECOVERED = "recovered"


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(bbox: list[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)


def _bbox_size(bbox: list[float]) -> float:
    return max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


class KalmanBoxTracker:
    """Constant-velocity Kalman filter on bbox center + size."""

    def __init__(self, bbox: list[float]):
        cx, cy = _center(bbox)
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        s = w * h
        r = w / h
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(7) * 10.0
        self.F = np.eye(7)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        self.Q = np.eye(7) * 0.05
        self.R = np.eye(4) * 1.0

    def predict(self, damp_velocity: float = 1.0) -> list[float]:
        if damp_velocity < 1.0:
            self.x[4:7] *= damp_velocity
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.to_bbox()

    def update(self, bbox: list[float]) -> None:
        cx, cy = _center(bbox)
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        z = np.array([cx, cy, w * h, w / h], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def to_bbox(self) -> list[float]:
        cx, cy, s, r = self.x[:4]
        s = max(1.0, s)
        r = max(1e-3, r)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-3)
        return [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]


@dataclass
class StablePlayer:
    player_id: str
    team: str = ""
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    appearance: Optional[np.ndarray] = None
    jersey_color: Optional[list[int]] = None
    state: TrackState = TrackState.TRACKED
    hits: int = 0
    time_since_update: int = 0
    last_temp_track_id: Optional[int] = None
    kalman: Optional[KalmanBoxTracker] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    init_appearance: Optional[np.ndarray] = None

    def ensure_kalman(self) -> KalmanBoxTracker:
        if self.kalman is None:
            self.kalman = KalmanBoxTracker(self.bbox)
        return self.kalman


@dataclass
class MatchWeights:
    w_position: float = 0.30
    w_iou: float = 0.20
    w_appearance: float = 0.35
    w_color: float = 0.10
    max_cost: float = 1.35
    # Looser gate when re-acquiring after occlusion / lost
    max_cost_recover: float = 2.8
    position_norm_px: float = 280.0
    position_norm_recover_px: float = 420.0


class PlayerIdentityManager:
    """Fuse location / IoU / appearance into stable Lykon player IDs."""

    def __init__(
        self,
        players_init: list[dict[str, Any]] | None = None,
        *,
        max_lost_frames: int = 90,
        occluded_frames: int = 20,
        mode: str = "motion",
        allow_new_players: bool = False,
        max_players: int = 6,
        weights: MatchWeights | None = None,
        appearance: AppearanceReIDInterface | None = None,
        # Keep matching forever for human-init motion (A/B only)
        persistent_roster: bool | None = None,
    ):
        self.max_lost_frames = max_lost_frames
        self.occluded_frames = occluded_frames
        self.mode = mode
        self.max_players = max_players
        self.allow_new_players = bool(allow_new_players)
        self._human_initialized = bool(players_init)
        self._bootstrapped = bool(players_init)
        # 1v1 / human-init: never drop roster slots; always try to rematch A/B
        if persistent_roster is None:
            persistent_roster = bool(players_init) or mode == "motion"
        self.persistent_roster = bool(persistent_roster)
        self.weights = weights or MatchWeights()
        self.appearance = appearance or AppearanceReIDInterface()
        self.players: dict[str, StablePlayer] = {}
        if players_init:
            for p in players_init:
                self.register_initialized(p)

    def _next_auto_name(self) -> str:
        if self.max_players <= 2:
            for name in ("A", "B"):
                if name not in self.players:
                    return name
        i = 1
        while f"P{i}" in self.players:
            i += 1
        return f"P{i}"

    def _bootstrap_from_dets(
        self,
        detections: list[dict[str, Any]],
        det_apps: list[Optional[np.ndarray]],
        skip: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        skip = skip or set()
        ranked = sorted(
            [j for j in range(len(detections)) if j not in skip],
            key=lambda j: float(detections[j].get("detection_confidence") or 0.0),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for j in ranked:
            if len(self.players) >= self.max_players:
                break
            pid = self._next_auto_name()
            det = detections[j]
            self.register_initialized({
                "player_id": pid,
                "team": pid[0] if pid[:1] in ("A", "B") else "",
                "bbox": det["bbox_xyxy"],
                "appearance": det_apps[j].tolist() if det_apps[j] is not None else None,
            })
            out.append(self._emit(self.players[pid], det, TrackState.TRACKED))
        if self.players:
            self._bootstrapped = True
        return out

    def register_initialized(self, p: dict[str, Any]) -> StablePlayer:
        pid = str(p["player_id"])
        bbox = [float(x) for x in p.get("bbox") or p.get("bbox_xyxy") or [0, 0, 0, 0]]
        app = p.get("appearance")
        appearance = np.asarray(app, dtype=np.float32) if app is not None else None
        sp = StablePlayer(
            player_id=pid,
            team=str(p.get("team") or (pid[0] if pid else "")),
            bbox=bbox,
            appearance=appearance,
            init_appearance=appearance.copy() if appearance is not None else None,
            jersey_color=p.get("jersey_color"),
            state=TrackState.TRACKED,
            hits=1,
            time_since_update=0,
            kalman=KalmanBoxTracker(bbox) if any(bbox) else None,
        )
        self.players[pid] = sp
        return sp

    def _cost_threshold(self, player: StablePlayer) -> float:
        if player.state in (TrackState.OCCLUDED, TrackState.LOST) or player.time_since_update > 1:
            return self.weights.max_cost_recover
        return self.weights.max_cost

    def _pair_cost(
        self,
        player: StablePlayer,
        det: dict[str, Any],
        det_appearance: Optional[np.ndarray],
    ) -> float:
        w = self.weights
        recovering = player.state in (TrackState.OCCLUDED, TrackState.LOST) or player.time_since_update > 1
        pred_bbox = player.kalman.to_bbox() if player.kalman is not None else player.bbox
        det_bbox = det["bbox_xyxy"]
        pos_norm = w.position_norm_recover_px if recovering else w.position_norm_px
        pos_dist = float(np.linalg.norm(_center(pred_bbox) - _center(det_bbox))) / max(pos_norm, 1.0)
        iou = _iou(pred_bbox, det_bbox)

        app_ref = player.appearance if player.appearance is not None else player.init_appearance
        app_dist = 1.0
        if app_ref is not None and det_appearance is not None:
            app_dist = self.appearance.distance(app_ref, det_appearance)
            # Also compare to frozen init appearance (more stable after long occlusion)
            if player.init_appearance is not None:
                app_init = self.appearance.distance(player.init_appearance, det_appearance)
                app_dist = min(app_dist, app_init)

        color_dist = 0.0
        if app_ref is not None and det_appearance is not None:
            color_dist = color_histogram_distance(app_ref, det_appearance)

        size_ratio = abs(_bbox_size(pred_bbox) - _bbox_size(det_bbox)) / max(_bbox_size(pred_bbox), 1.0)

        # During recovery, IoU is often ~0 after separation — downweight it
        w_iou = 0.08 if recovering else w.w_iou
        w_pos = 0.25 if recovering else w.w_position
        w_app = 0.45 if recovering else w.w_appearance
        w_col = 0.12 if recovering else w.w_color

        cost = (
            w_pos * min(pos_dist, 4.0)
            + w_iou * (1.0 - iou)
            + w_app * app_dist
            + w_col * color_dist
            + 0.05 * min(size_ratio, 2.0)
        )

        if (
            player.last_temp_track_id is not None
            and det.get("temporary_track_id") is not None
            and int(player.last_temp_track_id) == int(det["temporary_track_id"])
        ):
            cost *= 0.65

        # Strong appearance agreement can override bad geometry after occlusion
        if recovering and app_dist < 0.25:
            cost *= 0.75
        return float(cost)

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Match detections to stable players; return enriched detections."""
        for sp in self.players.values():
            damp = 1.0
            # After occlusion, damp velocity so prediction does not fly away
            if sp.time_since_update >= 1:
                damp = 0.92
            if sp.time_since_update >= self.occluded_frames:
                damp = 0.75
            if sp.kalman is not None:
                sp.kalman.predict(damp_velocity=damp)
                sp.bbox = sp.kalman.to_bbox()
            sp.time_since_update += 1

        det_apps: list[Optional[np.ndarray]] = []
        for det in detections:
            if frame_bgr is not None:
                det_apps.append(self.appearance.extract(frame_bgr, det["bbox_xyxy"]))
            else:
                det_apps.append(None)

        player_ids = list(self.players.keys())
        if not player_ids:
            return self._bootstrap_from_dets(detections, det_apps)

        if len(detections) == 0:
            self._mark_missing()
            return []

        n_p, n_d = len(player_ids), len(detections)
        cost = np.full((n_p, n_d), fill_value=1e3, dtype=float)
        for i, pid in enumerate(player_ids):
            sp = self.players[pid]
            for j, det in enumerate(detections):
                cost[i, j] = self._pair_cost(sp, det, det_apps[j])

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_players: set[str] = set()
        assigned_dets: set[int] = set()
        matched: list[dict[str, Any]] = []

        for r, c in zip(row_ind, col_ind):
            pid = player_ids[r]
            sp = self.players[pid]
            thresh = self._cost_threshold(sp)
            if cost[r, c] > thresh:
                continue
            det = detections[c]
            prev_center = _center(sp.bbox)
            new_state = TrackState.TRACKED
            if sp.time_since_update > 1 or sp.state in (TrackState.OCCLUDED, TrackState.LOST):
                new_state = TrackState.RECOVERED
            self._apply_detection(sp, det, det_apps[c], new_state)
            sp.velocity = _center(sp.bbox) - prev_center
            assigned_players.add(pid)
            assigned_dets.add(c)
            matched.append(self._emit(sp, det, new_state))

        # Second pass: greedy rematch remaining lost/occluded players to leftover dets
        unmatched_players = [pid for pid in player_ids if pid not in assigned_players]
        unmatched_dets = [j for j in range(n_d) if j not in assigned_dets]
        if unmatched_players and unmatched_dets:
            candidates = []
            for pid in unmatched_players:
                sp = self.players[pid]
                for j in unmatched_dets:
                    cval = self._pair_cost(sp, detections[j], det_apps[j])
                    if cval <= self.weights.max_cost_recover:
                        candidates.append((cval, pid, j))
            candidates.sort(key=lambda x: x[0])
            used_p: set[str] = set()
            used_d: set[int] = set()
            for cval, pid, j in candidates:
                if pid in used_p or j in used_d:
                    continue
                sp = self.players[pid]
                det = detections[j]
                prev_center = _center(sp.bbox)
                self._apply_detection(sp, det, det_apps[j], TrackState.RECOVERED)
                sp.velocity = _center(sp.bbox) - prev_center
                matched.append(self._emit(sp, det, TrackState.RECOVERED))
                used_p.add(pid)
                used_d.add(j)
                assigned_players.add(pid)
                assigned_dets.add(j)

        # Unmatched players → occluded / lost (but keep slot alive)
        for pid in player_ids:
            if pid in assigned_players:
                continue
            sp = self.players[pid]
            if sp.time_since_update <= self.occluded_frames:
                sp.state = TrackState.OCCLUDED
            else:
                sp.state = TrackState.LOST
            # Non-persistent mode can freeze after max_lost_frames (still keep object)
            if (not self.persistent_roster) and sp.time_since_update > self.max_lost_frames:
                sp.state = TrackState.LOST

        if (not self._human_initialized) and len(self.players) < self.max_players:
            matched.extend(self._bootstrap_from_dets(detections, det_apps, skip=assigned_dets))

        return matched

    def _apply_detection(
        self,
        sp: StablePlayer,
        det: dict[str, Any],
        appearance: Optional[np.ndarray],
        state: TrackState,
    ) -> None:
        bbox = [float(x) for x in det["bbox_xyxy"]]
        sp.bbox = bbox
        sp.ensure_kalman().update(bbox)
        sp.hits += 1
        sp.time_since_update = 0
        sp.state = state
        if det.get("temporary_track_id") is not None:
            sp.last_temp_track_id = int(det["temporary_track_id"])
        if appearance is not None:
            if sp.appearance is None:
                sp.appearance = appearance
                if sp.init_appearance is None:
                    sp.init_appearance = appearance.copy()
            else:
                # Slower EMA after recovery so one bad frame doesn't wipe identity
                alpha = 0.08 if state == TrackState.RECOVERED else 0.15
                sp.appearance = (1.0 - alpha) * sp.appearance + alpha * appearance

    def _mark_missing(self) -> None:
        for sp in self.players.values():
            if sp.time_since_update <= self.occluded_frames:
                sp.state = TrackState.OCCLUDED
            else:
                sp.state = TrackState.LOST

    def _emit(self, sp: StablePlayer, det: dict[str, Any], state: TrackState) -> dict[str, Any]:
        out = dict(det)
        out["stable_player_id"] = sp.player_id
        out["team"] = sp.team
        out["tracking_state"] = state.value
        out["predicted_bbox"] = sp.kalman.to_bbox() if sp.kalman is not None else sp.bbox
        return out

    def predicted_missing(self) -> list[dict[str, Any]]:
        """Emit Kalman-predicted placeholders for occluded/lost players."""
        out = []
        for sp in self.players.values():
            if sp.state not in (TrackState.OCCLUDED, TrackState.LOST):
                continue
            if not self.persistent_roster and sp.time_since_update > self.max_lost_frames:
                continue
            bbox = sp.kalman.to_bbox() if sp.kalman is not None else sp.bbox
            out.append({
                "stable_player_id": sp.player_id,
                "team": sp.team,
                "bbox_xyxy": bbox,
                "tracking_state": sp.state.value,
                "temporary_track_id": sp.last_temp_track_id,
                "detection_confidence": 0.0,
                "keypoints": None,
                "predicted": True,
                "pixel_foot_point": [(bbox[0] + bbox[2]) * 0.5, bbox[3]],
            })
        return out
