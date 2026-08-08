from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
from lykon.pipeline.baseline import run_baseline

st.set_page_config(page_title="Lykon Vision MVP", page_icon="🏀", layout="wide")
st.markdown("# LYKON")
st.caption("HONOR THE INSTINCT — Video Intelligence MVP")
st.info("This demo is video-first. Wearable/UWB fusion is intentionally not required yet.")

uploaded = st.file_uploader("Upload a short basketball video (MP4/MOV)", type=["mp4", "mov", "m4v", "avi"])

if uploaded:
    session = Path(tempfile.mkdtemp(prefix="lykon_streamlit_"))
    video_path = session / uploaded.name
    video_path.write_bytes(uploaded.read())
    st.video(str(video_path))

    if st.button("Run Lykon baseline", type="primary"):
        with st.spinner("Running player tracking, pose, metrics and 3D replay..."):
            summary = run_baseline(str(video_path), str(session / "output"), court_config=None, config_path=str(ROOT / "configs/default.yaml"))
        st.success("Done")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tracked video")
            st.video(summary["outputs"]["tracked_overlay"])
        with col2:
            st.subheader("3D replay baseline")
            st.video(summary["outputs"]["pseudo3d_replay"])
        st.subheader("Run summary")
        st.json(summary)
        metrics_path = Path(summary["outputs"]["metrics"])
        if metrics_path.exists():
            st.subheader("Metrics")
            st.json(json.loads(metrics_path.read_text()))
        events_path = Path(summary["outputs"]["events"])
        if events_path.exists():
            st.subheader("Shot-motion candidates")
            st.json(json.loads(events_path.read_text()))
        st.warning("Without court calibration, court-distance/tactical metrics are limited. Calibrate a fixed camera before the Hangzhou demo.")
