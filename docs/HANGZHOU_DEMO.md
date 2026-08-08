# Hangzhou 1v1 Demo Capture Protocol

## Public promise
One sideline camera + Lykon wearables, 1v1 half court, approximately three minutes, followed by a digital 3D replay experience.

## What this repo should deliver for the demo
- persistent player IDs for two players
- 2D body pose
- half-court XY trajectories after calibration
- movement metrics
- shot-motion candidate timestamps
- 3D skeleton / WHAM reconstruction
- replay-ready JSON

## Camera
Preferred:
- fixed tripod / light stand
- sideline, around half-court center
- elevated roughly 2.5–4 m if venue allows
- oblique downward view rather than straight top-down
- full half court and both players remain visible
- 4K if available
- 60 fps minimum preferred; 120 fps is useful for release analysis if lighting allows
- disable digital stabilization if it changes crop dynamically
- lock focus/exposure if possible
- avoid auto panning or zooming

## Before the match
1. Mount camera and do not move it afterwards.
2. Record 10 seconds with a person walking to all four court corners.
3. Run `scripts/calibrate_court.py`.
4. Record a 10–20 s mini 1v1 test.
5. Run the full pipeline before the official session.
6. Confirm both output videos can be played.

## Data collection
Keep:
- original camera file
- exact camera model / fps / resolution
- game start/end wall-clock time
- wearable raw files and device IDs
- player mapping (P1/P2)
- manual score log
- any known shot timestamps if a staff member can mark them

## R&D backup
Use a second camera from another angle if possible. It is not required for the public product story; it is an insurance and validation camera for algorithm development.
