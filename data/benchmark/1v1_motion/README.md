# 1v1 Motion Benchmark

Basket-side / rear **fixed** camera. Players appear large in frame.

## Priorities
- Stable Player A / Player B identity (near-zero stable ID switches)
- Pose continuity under occlusion
- Downstream 3D reconstruction (WHAM / SMPL)

## Not the right metric
Do not judge this scene primarily by tactical spacing or 6-player coverage.

## Layout
```text
data/benchmark/1v1_motion/
  test.mp4              # place your clip here
  README.md
```

## Suggested setup
```bash
python scripts/init_players.py \
  --video data/benchmark/1v1_motion/test.mp4 \
  --num-players 2 \
  --names A,B \
  --output configs/players_1v1.json

python scripts/calibrate_court_roi.py \
  --video data/benchmark/1v1_motion/test.mp4 \
  --output configs/court_roi_1v1.json

python scripts/calibrate_court.py \
  --video data/benchmark/1v1_motion/test.mp4 \
  --output configs/court_1v1.json

python scripts/run_pipeline.py \
  --video data/benchmark/1v1_motion/test.mp4 \
  --mode motion \
  --player-init configs/players_1v1.json \
  --court-config configs/court_1v1.json \
  --court-roi configs/court_roi_1v1.json \
  --config configs/motion_1v1.yaml \
  --output data/output/1v1_run
```
