# 3v3 Tactical Benchmark

Midcourt / sideline **fixed** oblique camera. Multiple small players + spectators.

## Priorities
- Detect all 6 active players
- Stable A1–A3 / B1–B3 identities (minimize stable ID switches)
- Court ROI filtering of coaches/spectators
- Court XY trajectories + top-down viz

## Not the right metric
Do not judge this scene primarily by close-up pose / 3D mesh quality.

## Layout
```text
data/benchmark/3v3_tactical/
  test.mp4              # place your clip here
  README.md
```

## Suggested setup
```bash
python scripts/init_players.py \
  --video data/benchmark/3v3_tactical/test.mp4 \
  --num-players 6 \
  --names A1,A2,A3,B1,B2,B3 \
  --output configs/players_3v3.json

python scripts/calibrate_court_roi.py \
  --video data/benchmark/3v3_tactical/test.mp4 \
  --output configs/court_roi_3v3.json

python scripts/calibrate_court.py \
  --video data/benchmark/3v3_tactical/test.mp4 \
  --output configs/court_3v3.json

python scripts/run_pipeline.py \
  --video data/benchmark/3v3_tactical/test.mp4 \
  --mode tactical \
  --player-init configs/players_3v3.json \
  --court-config configs/court_3v3.json \
  --court-roi configs/court_roi_3v3.json \
  --config configs/tactical_3v3.yaml \
  --output data/output/3v3_run
```
