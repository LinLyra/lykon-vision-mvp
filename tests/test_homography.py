import numpy as np
from lykon.court.homography import compute_homography, image_to_court


def test_corner_mapping():
    cfg = {
        "image_points": [[0, 100], [100, 100], [100, 0], [0, 0]],
        "court_points_m": [[0, 0], [15, 0], [15, 14], [0, 14]],
    }
    H = compute_homography(cfg)
    x, y = image_to_court((100, 0), H)
    assert abs(x - 15) < 1e-4
    assert abs(y - 14) < 1e-4
