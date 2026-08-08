# Qoder: start here

Goal for today: get **10–20 seconds of basketball video -> tracked players -> pseudo-3D replay**.

Run exactly in this order:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Put a short clip at:

```text
data/input/test.mp4
```

Run:

```bash
python scripts/run_pipeline.py --video data/input/test.mp4 --output data/output/first_run
```

Open:

```text
data/output/first_run/tracked_overlay.mp4
data/output/first_run/pseudo3d_replay.mp4
```

If those two files exist, the first milestone is done.

Next run court calibration:

```bash
python scripts/calibrate_court.py --video data/input/test.mp4 --output configs/my_court.json
```

Then rerun with:

```bash
python scripts/run_pipeline.py --video data/input/test.mp4 --court-config configs/my_court.json --output data/output/calibrated_run
```

Only after the baseline is healthy, try WHAM:

```bash
bash scripts/setup_wham.sh
python scripts/run_wham.py --video data/input/test.mp4 --wham-dir external/WHAM --output data/output/wham_run
```
