from lykon.court.roi import filter_detections_by_roi, is_on_court, point_in_polygon, save_court_roi
import numpy as np
from pathlib import Path


def test_point_in_polygon():
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert point_in_polygon((50, 50), poly)
    assert not point_in_polygon((150, 50), poly)


def test_filter_detections_by_roi(tmp_path: Path):
    roi = save_court_roi([[0, 0], [200, 0], [200, 200], [0, 200]], tmp_path / "roi.json")
    dets = [
        {"bbox_xyxy": [40, 40, 80, 120], "pixel_foot_point": [60, 120]},   # inside
        {"bbox_xyxy": [300, 40, 340, 120], "pixel_foot_point": [320, 120]},  # outside
    ]
    active, ignored = filter_detections_by_roi(dets, roi)
    assert len(active) == 1
    assert len(ignored) == 1
    assert is_on_court([60, 120], roi)
    assert not is_on_court([320, 120], roi)


def test_missing_roi_keeps_all():
    dets = [{"bbox_xyxy": [0, 0, 10, 10], "pixel_foot_point": [5, 10]}]
    active, ignored = filter_detections_by_roi(dets, None)
    assert len(active) == 1
    assert ignored == []
