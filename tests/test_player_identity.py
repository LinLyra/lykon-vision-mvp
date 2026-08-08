import numpy as np

from lykon.tracking.player_identity import PlayerIdentityManager, TrackState


def _det(tid, bbox, conf=0.9):
    return {
        "temporary_track_id": tid,
        "bbox_xyxy": bbox,
        "detection_confidence": conf,
        "keypoints": [[float(i), float(i), 0.9] for i in range(17)],
        "pixel_foot_point": [(bbox[0] + bbox[2]) / 2, bbox[3]],
    }


def test_stable_ids_survive_temp_id_change():
    init = [
        {"player_id": "A", "team": "A", "bbox": [100, 100, 160, 260]},
        {"player_id": "B", "team": "B", "bbox": [300, 100, 360, 260]},
    ]
    mgr = PlayerIdentityManager(init, mode="motion", max_lost_frames=20)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Paint jersey-ish colors
    frame[120:200, 110:150] = (0, 0, 220)
    frame[120:200, 310:350] = (220, 0, 0)

    m1 = mgr.update([_det(3, [102, 102, 162, 262]), _det(8, [302, 102, 362, 262])], frame)
    assert {x["stable_player_id"] for x in m1} == {"A", "B"}

    # Temporary tracker IDs swap / change — stable IDs must hold
    m2 = mgr.update([_det(11, [104, 104, 164, 264]), _det(19, [304, 104, 364, 264])], frame)
    ids = {x["stable_player_id"] for x in m2}
    assert ids == {"A", "B"}
    assert all(x["stable_player_id"] in ("A", "B") for x in m2)


def test_motion_mode_does_not_spawn_third_player():
    init = [
        {"player_id": "A", "team": "A", "bbox": [100, 100, 160, 260]},
        {"player_id": "B", "team": "B", "bbox": [300, 100, 360, 260]},
    ]
    mgr = PlayerIdentityManager(init, mode="motion", allow_new_players=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [
        _det(1, [100, 100, 160, 260]),
        _det(2, [300, 100, 360, 260]),
        _det(3, [500, 100, 560, 260]),  # spectator-like
    ]
    matched = mgr.update(dets, frame)
    assert len(matched) <= 2
    assert "C" not in {m["stable_player_id"] for m in matched}


def test_occlusion_state_machine():
    init = [{"player_id": "A", "team": "A", "bbox": [100, 100, 160, 260]}]
    mgr = PlayerIdentityManager(init, mode="motion", max_lost_frames=10, occluded_frames=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mgr.update([_det(1, [100, 100, 160, 260])], frame)
    for _ in range(2):
        mgr.update([], frame)
    assert mgr.players["A"].state == TrackState.OCCLUDED
    for _ in range(5):
        mgr.update([], frame)
    assert mgr.players["A"].state == TrackState.LOST
    out = mgr.update([_det(9, [105, 105, 165, 265])], frame)
    assert out[0]["stable_player_id"] == "A"
    assert out[0]["tracking_state"] in ("tracked", "recovered")


def test_occlusion_then_far_reacquire():
    """After occlusion, B should rematch even if IoU is ~0 (geometry drifted)."""
    init = [
        {"player_id": "A", "team": "A", "bbox": [100, 100, 160, 260],
         "appearance": [1.0] + [0.0] * 127},
        {"player_id": "B", "team": "B", "bbox": [300, 100, 360, 260],
         "appearance": [0.0] * 64 + [1.0] + [0.0] * 63},
    ]
    mgr = PlayerIdentityManager(init, mode="motion", max_lost_frames=60, occluded_frames=5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[120:200, 110:150] = (0, 0, 220)
    frame[120:200, 310:350] = (220, 80, 0)

    mgr.update([_det(1, [100, 100, 160, 260]), _det(2, [300, 100, 360, 260])], frame)
    for _ in range(12):
        out = mgr.update([_det(1, [105, 105, 165, 265])], frame)
        assert "A" in {x["stable_player_id"] for x in out}

    preds = mgr.predicted_missing()
    assert any(p["stable_player_id"] == "B" for p in preds)

    out = mgr.update(
        [_det(1, [110, 110, 170, 270]), _det(9, [340, 130, 400, 290])],
        frame,
    )
    ids = {x["stable_player_id"] for x in out}
    assert ids == {"A", "B"}
    assert "C" not in ids
