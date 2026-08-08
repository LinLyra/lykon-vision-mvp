"""Sports-ball detection via Ultralytics YOLO (COCO class 32).

Not pose-based. Recall-first defaults for small / fast basketball.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# COCO class id for sports ball
SPORTS_BALL_CLASS = 32


@dataclass
class BallDetection:
    bbox_xyxy: list[float]
    center_xy: list[float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": [float(x) for x in self.bbox_xyxy],
            "center_xy": [float(x) for x in self.center_xy],
            "confidence": float(self.confidence),
        }


class BallDetector:
    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        *,
        conf: float = 0.10,
        iou: float = 0.5,
        imgsz: int = 1280,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.model_name = model_name

    def detect(self, frame_bgr: np.ndarray) -> list[BallDetection]:
        kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "classes": [SPORTS_BALL_CLASS],
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        out: list[BallDetection] = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            out.append(
                BallDetection(
                    bbox_xyxy=[x1, y1, x2, y2],
                    center_xy=[(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                    confidence=conf,
                )
            )
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
