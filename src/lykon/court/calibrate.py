from __future__ import annotations

import json
from pathlib import Path
import cv2


def calibrate_from_video(video_path: str, output_path: str, width_m: float = 15.0, depth_m: float = 14.0) -> dict:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")

    points: list[tuple[int, int]] = []
    display = frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    window = "Lykon Court Calibration: click near-left, near-right, far-right, far-left; ESC to cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while len(points) < 4:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Calibration cancelled")

    cv2.destroyAllWindows()
    config = {
        "image_points": [[float(x), float(y)] for x, y in points],
        "court_points_m": [[0.0, 0.0], [width_m, 0.0], [width_m, depth_m], [0.0, depth_m]],
        "note": "Half-court plane calibration for Lykon video analytics.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
