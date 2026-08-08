# Lykon Vision MVP

**HONOR THE INSTINCT**

A video-first basketball intelligence prototype for Lykon. This repository deliberately does **not** depend on the wearable/UWB stack yet. It is designed to get the Hangzhou demo pipeline running first:

`basketball video -> player tracking -> 2D pose -> court coordinates -> events/tactics -> 3D replay`

## What works now

### Fast baseline (recommended first)
- Player detection + persistent tracking using Ultralytics pose models.
- 17-keypoint 2D pose extraction per tracked player.
- Manual 4-point half-court calibration and homography to court coordinates.
- Player trajectories, speed, distance, spacing and 1v1 separation metrics.
- Simple shot-motion candidate detector (heuristic, not production shot recognition).
- A lightweight pseudo-3D skeleton replay that gives you an immediate animated 3D result without SMPL registration.
- Streamlit demo UI to upload a video and inspect outputs.

### High-fidelity 3D path
- Optional WHAM adapter that runs the official WHAM repository for world-grounded SMPL reconstruction.
- WHAM's own visualization can be copied into the Lykon output directory.
- A future Blender/Unity retarget layer can consume SMPL motion; this repo keeps that interface separate from analytics.

## Why two paths?
WHAM is a much better long-term 3D reconstruction route, but it requires CUDA-class compute plus SMPL/SMPLify model access and its own dependency stack. The baseline path makes sure you can validate video capture, tracking, court calibration, trajectories and replay today.

---

## 1. Environment

Recommended: Python 3.11 (supported range: `>=3.10,<3.12`).

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\\Scripts\\activate  # Windows

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

For NVIDIA CUDA machines, install the correct PyTorch build before `requirements.txt` if needed.

## 2. Put a test video here

```text
data/input/test.mp4
```

For the fastest first test use a **10–20 second**, fixed-camera clip with one or two full-body players visible.

## 3. Run the complete baseline

```bash
python scripts/run_pipeline.py \
  --video data/input/test.mp4 \
  --output data/output/test_run
```

Outputs:

```text
data/output/test_run/
├── tracks_pose.json
├── tracked_overlay.mp4
├── trajectories.json
├── metrics.json
├── events.json
├── pseudo3d.json
├── pseudo3d_replay.mp4
└── run_summary.json
```

If no court calibration is supplied, pixel-space trajectories still work. For tactical court coordinates, run calibration first.

## 4. Calibrate the court

Choose four image points corresponding to the four corners of the visible half court in this order:

1. near-left
2. near-right
3. far-right
4. far-left

Run:

```bash
python scripts/calibrate_court.py \
  --video data/input/test.mp4 \
  --output configs/court_hangzhou.json
```

Then:

```bash
python scripts/run_pipeline.py \
  --video data/input/test.mp4 \
  --court-config configs/court_hangzhou.json \
  --output data/output/test_run
```

The default metric court plane is half court: 15 m wide x 14 m deep. Adjust in the config if your visible area differs.

## 5. Launch the web demo

```bash
streamlit run app/streamlit_app.py
```

The web demo lets you upload a video, run the baseline and inspect:
- tracked video
- pseudo-3D replay
- distance / speed / spacing metrics
- detected shot-motion candidates
- JSON outputs

## 6. Upgrade to WHAM

WHAM is kept as an external research dependency rather than vendored into this repo.

```bash
bash scripts/setup_wham.sh
```

Then follow the WHAM/SMPL registration instructions inside `external/WHAM` to fetch required body models/checkpoints.

Run:

```bash
python scripts/run_wham.py \
  --video data/input/test.mp4 \
  --wham-dir external/WHAM \
  --output data/output/wham_test
```

The adapter invokes the official WHAM demo. If WHAM creates a visualization video, the adapter copies it to the Lykon output folder and records where additional result files were produced.

### Important
WHAM's repository requires SMPL/SMPLify assets whose licensing/registration is managed by their original providers. This repo does not redistribute those files.

---

# Pipeline architecture

```text
VIDEO
  |
  +--> Person detection (YOLO pose model, temporary track_id only)
  |
  +--> Court ROI filter (drop spectators / coaches)
  |
  +--> Human player initialization (A/B or A1..B3)
  |
  +--> Short-term MOT + appearance ReID / HSV
  |       |
  |       +--> stable_player_id  (≠ temporary_track_id)
  |       +--> tracking_state: tracked|occluded|lost|recovered
  |
  +--> Pose estimate on tracked players
  |       |
  |       +--> pose_valid / pose_source (detected|interpolated|missing)
  |       +--> short-gap pose recovery
  |
  +--> Court calibration (homography)
  |       |
  |       +--> court XY trajectories + top-down viz
  |
  +--> Basketball analytics + pseudo-3D / WHAM adapter
```

## Two benchmarks (different goals)

| Scene | Folder | Camera | Optimize for |
|-------|--------|--------|--------------|
| 1v1 motion | `data/benchmark/1v1_motion/` | basket rear/side | pose + stable A/B + 3D |
| 3v3 tactical | `data/benchmark/3v3_tactical/` | midcourt/sideline | identity + court XY + tactics |

Do **not** score both scenes with the same checklist.

## Human-initialized identity (Hangzhou demo)

```bash
python scripts/init_players.py --video ... --num-players 6 --output configs/players_3v3.json
python scripts/calibrate_court_roi.py --video ... --output configs/court_roi_3v3.json
```

This is intentional product design: AI tracks the players you named, it does not guess who is in the game.

# What to add before the Hangzhou event

Priority order:

1. Validate the camera angle with actual venue-like footage.
2. Keep both players full-body in frame.
3. Lock exposure/focus and use 60 fps or higher if possible.
4. Calibrate the half court once the camera is mounted.
5. Collect one short test clip and run the full pipeline before the event starts.
6. Record a second backup camera for R&D even if the public demo is single-camera.
7. Keep raw videos and wearable timestamps; do not overwrite them after rendering.

# Tactical analytics roadmap

Current repo ships reliable geometry first and leaves advanced basketball semantics as explicit modules:

- `player_tracking`: who is where in the frame
- `court_mapping`: where each player is on court
- `motion_metrics`: distance, speed, acceleration, spacing
- `shot_candidates`: possible shot motion windows
- future `ball_tracking`: ball trajectory and shot outcome
- future `possession`: who controls the ball
- future `play_events`: drive, screen, handoff, cut, closeout, help rotation
- future `tactics`: spacing quality, drive lane opening, defensive gap, switch/help events

Do not label these advanced events as production-accurate until validated on basketball-specific annotated data.

# Data schema

Each tracked pose sample uses a **unified** schema (`keypoints` always `[x, y, confidence]`):

```json
{
  "frame_idx": 120,
  "time_s": 2.0,
  "temporary_track_id": 37,
  "stable_player_id": "A1",
  "bbox_xyxy": [100, 80, 280, 620],
  "detection_confidence": 0.91,
  "keypoints": [[x, y, confidence]],
  "pose_valid": true,
  "pose_source": "detected",
  "pixel_foot_point": [190, 620],
  "court_xy_m": [5.4, 8.2],
  "tracking_state": "tracked"
}
```

Never treat YOLO `temporary_track_id` as a permanent player identity.

# Product boundary

This repository is the **video intelligence subsystem**, not the full Lykon stack.

Future interfaces:

```text
wearable IMU ----> motion binding ----\
                                     +--> unified game state --> replay
Court Hub/UWB ---> positioning -------/
video -----------> visual semantics --/
```

That keeps today's prototype usable while preserving the future sensor-fusion architecture.
