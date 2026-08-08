from lykon.schema import coerce_sample, make_player_sample, normalize_keypoints, pose_valid_from_keypoints


def test_keypoints_always_xyz_conf():
    a = normalize_keypoints([[1, 2], [3, 4]], [0.9, 0.8])
    assert a[0] == [1.0, 2.0, 0.9]
    assert a[1] == [3.0, 4.0, 0.8]
    assert len(a) == 17

    b = normalize_keypoints([[1, 2, 0.5]])
    assert b[0] == [1.0, 2.0, 0.5]


def test_make_player_sample_schema():
    s = make_player_sample(
        frame_idx=3,
        time_s=0.1,
        stable_player_id="A1",
        bbox_xyxy=[10, 20, 50, 120],
        detection_confidence=0.8,
        keypoints=[[i, i, 0.9] for i in range(17)],
        temporary_track_id=7,
    )
    assert s["stable_player_id"] == "A1"
    assert s["temporary_track_id"] == 7
    assert s["track_id"] == "A1"
    assert len(s["keypoints"][0]) == 3
    assert s["pose_valid"] is True
    assert s["pose_source"] == "detected"
    assert "pixel_foot_point" in s
    assert s["bbox_xyxy"] == s["bbox"]


def test_coerce_legacy_sample():
    legacy = {
        "frame_idx": 1,
        "timestamp_s": 0.03,
        "track_id": 9,
        "bbox": [0, 0, 10, 20],
        "keypoints": [[1, 2] for _ in range(17)],
        "keypoint_conf": [0.5] * 17,
    }
    s = coerce_sample(legacy)
    assert s["keypoints"][0][2] == 0.5
    assert s["frame_idx"] == 1
    assert s["time_s"] == 0.03


def test_pose_valid():
    good = [[1, 1, 0.9] for _ in range(17)]
    bad = [[0, 0, 0.0] for _ in range(17)]
    assert pose_valid_from_keypoints(good) is True
    assert pose_valid_from_keypoints(bad) is False
