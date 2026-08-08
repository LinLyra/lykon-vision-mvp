"""Simple temporal basketball tracker (single ball).

Handles missed detections with velocity prediction and short-gap interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from lykon.ball.detector import BallDetection

BallSource = Literal["detected", "predicted", "interpolated"]


@dataclass
class BallTrackState:
    center_xy: np.ndarray
    velocity_xy: np.ndarray
    bbox_xyxy: np.ndarray | None = None
    confidence: float = 0.0
    age: int = 0
    time_since_update: int = 0
    hits: int = 0


@dataclass
class BallFrame:
    frame_idx: int
    time_s: float
    detected: bool
    bbox_xyxy: list[float] | None
    center_xy: list[float] | None
    confidence: float
    source: BallSource

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "time_s": float(self.time_s),
            "detected": bool(self.detected),
            "bbox_xyxy": self.bbox_xyxy,
            "center_xy": self.center_xy,
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass
class BallTracker:
    """Nearest-candidate association with displacement gate + short prediction."""

    max_lost_frames: int = 20
    max_displacement_px: float = 180.0
    interpolate_gap: int = 8
    velocity_ema: float = 0.6

    state: BallTrackState | None = None
    history: list[BallFrame] = field(default_factory=list)
    _pending_gap: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.state = None
        self.history = []
        self._pending_gap = []

    def _predict_center(self) -> np.ndarray | None:
        if self.state is None:
            return None
        return self.state.center_xy + self.state.velocity_xy

    def _match(self, detections: list[BallDetection]) -> BallDetection | None:
        if not detections:
            return None
        if self.state is None:
            return detections[0]  # highest confidence (already sorted)

        pred = self._predict_center()
        assert pred is not None
        best = None
        best_dist = float("inf")
        for det in detections:
            c = np.asarray(det.center_xy, dtype=float)
            dist = float(np.linalg.norm(c - pred))
            # Also allow matching to last known center for low-velocity cases
            dist_last = float(np.linalg.norm(c - self.state.center_xy))
            d = min(dist, dist_last)
            # Soft preference for higher confidence when distances are close
            score = d - 15.0 * det.confidence
            if d <= self.max_displacement_px and score < best_dist:
                best_dist = score
                best = det
        return best

    def _apply_detection(self, det: BallDetection) -> None:
        c = np.asarray(det.center_xy, dtype=float)
        bbox = np.asarray(det.bbox_xyxy, dtype=float)
        if self.state is None:
            self.state = BallTrackState(
                center_xy=c,
                velocity_xy=np.zeros(2, dtype=float),
                bbox_xyxy=bbox,
                confidence=det.confidence,
                age=1,
                time_since_update=0,
                hits=1,
            )
            return
        vel = c - self.state.center_xy
        self.state.velocity_xy = (
            self.velocity_ema * self.state.velocity_xy + (1.0 - self.velocity_ema) * vel
        )
        self.state.center_xy = c
        self.state.bbox_xyxy = bbox
        self.state.confidence = float(det.confidence)
        self.state.time_since_update = 0
        self.state.hits += 1
        self.state.age += 1

    def update(
        self,
        frame_idx: int,
        time_s: float,
        detections: list[BallDetection],
    ) -> BallFrame:
        matched = self._match(detections)

        if matched is not None:
            # If we had a short gap, mark pending predicted frames as interpolated bridge
            if self._pending_gap and len(self._pending_gap) <= self.interpolate_gap:
                for pending in self._pending_gap:
                    pending["source"] = "interpolated"
                    # linear interpolate centers toward current match
                self._finalize_gap_interpolation(matched.center_xy)
            self._pending_gap = []
            self._apply_detection(matched)
            frame = BallFrame(
                frame_idx=frame_idx,
                time_s=time_s,
                detected=True,
                bbox_xyxy=[float(x) for x in matched.bbox_xyxy],
                center_xy=[float(x) for x in matched.center_xy],
                confidence=float(matched.confidence),
                source="detected",
            )
            self.history.append(frame)
            return frame

        # No match
        if self.state is None:
            frame = BallFrame(
                frame_idx=frame_idx,
                time_s=time_s,
                detected=False,
                bbox_xyxy=None,
                center_xy=None,
                confidence=0.0,
                source="predicted",
            )
            self.history.append(frame)
            return frame

        self.state.time_since_update += 1
        self.state.age += 1
        if self.state.time_since_update > self.max_lost_frames:
            # Give up active track; keep last history entry as missing-like predicted null
            self.state = None
            self._pending_gap = []
            frame = BallFrame(
                frame_idx=frame_idx,
                time_s=time_s,
                detected=False,
                bbox_xyxy=None,
                center_xy=None,
                confidence=0.0,
                source="predicted",
            )
            self.history.append(frame)
            return frame

        pred = self._predict_center()
        assert pred is not None
        self.state.center_xy = pred
        # Expand last bbox around predicted center if available
        bbox = None
        if self.state.bbox_xyxy is not None:
            w = float(self.state.bbox_xyxy[2] - self.state.bbox_xyxy[0])
            h = float(self.state.bbox_xyxy[3] - self.state.bbox_xyxy[1])
            bbox = [
                float(pred[0] - w * 0.5),
                float(pred[1] - h * 0.5),
                float(pred[0] + w * 0.5),
                float(pred[1] + h * 0.5),
            ]
            self.state.bbox_xyxy = np.asarray(bbox, dtype=float)

        frame = BallFrame(
            frame_idx=frame_idx,
            time_s=time_s,
            detected=False,
            bbox_xyxy=bbox,
            center_xy=[float(pred[0]), float(pred[1])],
            confidence=0.0,
            source="predicted",
        )
        self.history.append(frame)
        self._pending_gap.append(
            {
                "index": len(self.history) - 1,
                "center": [float(pred[0]), float(pred[1])],
            }
        )
        # If gap already too long, leave as predicted (not interpolated)
        if len(self._pending_gap) > self.interpolate_gap:
            self._pending_gap = []
        return frame

    def _finalize_gap_interpolation(self, end_center: list[float] | np.ndarray) -> None:
        if not self._pending_gap:
            return
        start_center = None
        # find last detected before gap
        first_idx = self._pending_gap[0]["index"]
        for i in range(first_idx - 1, -1, -1):
            if self.history[i].source == "detected" and self.history[i].center_xy is not None:
                start_center = np.asarray(self.history[i].center_xy, dtype=float)
                break
        if start_center is None:
            return
        end = np.asarray(end_center, dtype=float)
        n = len(self._pending_gap)
        for k, pending in enumerate(self._pending_gap, start=1):
            alpha = k / (n + 1)
            c = (1.0 - alpha) * start_center + alpha * end
            idx = pending["index"]
            fr = self.history[idx]
            self.history[idx] = BallFrame(
                frame_idx=fr.frame_idx,
                time_s=fr.time_s,
                detected=False,
                bbox_xyxy=fr.bbox_xyxy,
                center_xy=[float(c[0]), float(c[1])],
                confidence=0.0,
                source="interpolated",
            )
