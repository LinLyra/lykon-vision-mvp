from lykon.pose.recovery import recover_poses
from lykon.schema import make_player_sample


def _sample(frame, confs, pid="A"):
    kps = [[10.0 + frame, 20.0 + i, c] for i, c in enumerate(confs)]
    return make_player_sample(
        frame_idx=frame,
        time_s=frame / 30.0,
        stable_player_id=pid,
        bbox_xyxy=[10, 10, 50, 100],
        detection_confidence=0.9,
        keypoints=kps,
        temporary_track_id=1,
        pose_source="detected",
    )


def test_short_gap_interpolation():
    confs = [0.9] * 17
    samples = [_sample(0, confs), _sample(5, confs)]
    # Remove middle — recovery should fill frames 1..4
    out = recover_poses(samples, max_gap=8)
    frames = {s["frame_idx"] for s in out if s["stable_player_id"] == "A"}
    assert 0 in frames and 5 in frames
    mid = [s for s in out if s["frame_idx"] == 2][0]
    assert mid["pose_source"] == "interpolated"
    assert len(mid["keypoints"][0]) == 3
    # Interpolated confidence must not be fake-high
    assert mid["keypoints"][0][2] < 0.9


def test_long_gap_not_forged():
    confs = [0.9] * 17
    samples = [_sample(0, confs), _sample(20, confs)]
    out = recover_poses(samples, max_gap=5)
    frames = {s["frame_idx"] for s in out}
    # Should not invent a dense bridge across a long gap
    assert 10 not in frames
