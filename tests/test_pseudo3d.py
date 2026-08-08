from lykon.reconstruct.pseudo3d import build_pseudo3d


def test_build_pseudo3d():
    kps = [[float(i), float(i * 2), 0.9] for i in range(17)]
    sample = {
        "frame_idx": 0,
        "frame": 0,
        "time_s": 0.0,
        "stable_player_id": "A",
        "track_id": "A",
        "bbox_xyxy": [0, 0, 100, 200],
        "keypoints": kps,
        "pose_valid": True,
        "pose_source": "detected",
    }
    out = build_pseudo3d([sample])
    assert "0" in out["frames"]
    assert len(out["frames"]["0"][0]["joints_xyz_conf"]) == 17
    assert out["frames"]["0"][0]["stable_player_id"] == "A"
