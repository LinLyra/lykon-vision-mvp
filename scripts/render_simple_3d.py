#!/usr/bin/env python3
"""Render a simple watchable pseudo-3D skeleton replay from tracks_pose.json.

Does NOT re-run YOLO / pose / tracking. Reads existing pose JSON only.
Heuristic depth only — not metric 3D, WHAM, or SMPL.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# COCO-17 topology
SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]

PLAYER_COLORS = {
    "A": "#2EC4FF",
    "B": "#FF9F1C",
}
FALLBACK_COLORS = ["#2EC4FF", "#FF9F1C", "#A3FF12", "#FF4D8D"]

LEFT_JOINTS = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}
CONF_THRESH = 0.2


def _pid(sample: dict[str, Any]) -> str:
    return str(sample.get("stable_player_id") or sample.get("track_id") or "?")


def _frame_idx(sample: dict[str, Any]) -> int:
    return int(sample.get("frame_idx", sample.get("frame", 0)))


def _time_s(sample: dict[str, Any]) -> float:
    if "time_s" in sample:
        return float(sample["time_s"])
    if "timestamp_s" in sample:
        return float(sample["timestamp_s"])
    return 0.0


def _keypoints(sample: dict[str, Any]) -> np.ndarray:
    kps = np.asarray(sample.get("keypoints") or [], dtype=float)
    if kps.ndim != 2:
        kps = np.zeros((17, 3), dtype=float)
    if kps.shape[1] == 2:
        conf = np.ones((len(kps), 1), dtype=float)
        kps = np.concatenate([kps, conf], axis=1)
    if len(kps) < 17:
        pad = np.zeros((17 - len(kps), 3), dtype=float)
        kps = np.vstack([kps, pad])
    return kps[:17]


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth along axis 0. values: [T, ...]"""
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    flat = values.reshape(len(values), -1).astype(float, copy=True)
    for c in range(flat.shape[1]):
        x = flat[:, c]
        xp = np.pad(x, (pad, pad), mode="edge")
        flat[:, c] = np.convolve(xp, kernel, mode="valid")
    return flat.reshape(values.shape)


def estimate_root_xy(kps: np.ndarray) -> np.ndarray:
    """Approximate ground contact / pelvis projection in image pixels."""
    conf = kps[:, 2]
    ankles = []
    for idx in (15, 16):
        if conf[idx] >= CONF_THRESH:
            ankles.append(kps[idx, :2])
    if ankles:
        return np.mean(ankles, axis=0)
    if conf[11] >= CONF_THRESH and conf[12] >= CONF_THRESH:
        return (kps[11, :2] + kps[12, :2]) * 0.5
    valid = conf >= CONF_THRESH
    if valid.any():
        return kps[valid, :2].mean(axis=0)
    return np.array([0.0, 0.0], dtype=float)


def pose2d_to_local3d(kps: np.ndarray, depth_scale: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Convert COCO-17 [x,y,conf] to local pseudo-3D joints [x,y,z] + confidence.

    Local frame:
      - origin near pelvis / visible center
      - +Y up
      - heuristic Z depth (left/right bias + lateral offset)
    Bone lengths are later stabilized temporally.
    """
    xy = kps[:, :2].astype(float)
    conf = kps[:, 2].astype(float)
    valid = conf >= CONF_THRESH
    joints = np.zeros((17, 3), dtype=float)
    if valid.sum() < 4:
        return joints, conf

    if conf[11] >= CONF_THRESH and conf[12] >= CONF_THRESH:
        center = (xy[11] + xy[12]) * 0.5
        shoulder = (xy[5] + xy[6]) * 0.5 if conf[5] >= CONF_THRESH and conf[6] >= CONF_THRESH else center
        torso = float(np.linalg.norm(shoulder - center))
    else:
        center = xy[valid].mean(axis=0)
        torso = float(np.ptp(xy[valid, 1]) * 0.25)
    scale = max(torso, 30.0)

    # normalized image coords → local body frame (Y up)
    lx = (xy[:, 0] - center[0]) / scale
    ly = -(xy[:, 1] - center[1]) / scale
    lz = np.zeros(17, dtype=float)

    for i in LEFT_JOINTS:
        lz[i] -= depth_scale * 0.10
    for i in RIGHT_JOINTS:
        lz[i] += depth_scale * 0.10
    # Mild depth from lateral position + arm/leg extension
    lz += depth_scale * 0.18 * lx
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if conf[hip] >= CONF_THRESH and conf[ankle] >= CONF_THRESH:
            # forward lean proxy from hip-ankle image foreshortening
            foreshort = 1.0 - min(1.0, abs(xy[ankle, 1] - xy[hip, 1]) / (scale * 2.2))
            lz[knee] += depth_scale * 0.12 * foreshort
            lz[ankle] += depth_scale * 0.18 * foreshort
    for sh, el, wr in ((5, 7, 9), (6, 8, 10)):
        if conf[sh] >= CONF_THRESH and conf[wr] >= CONF_THRESH:
            reach = float(np.linalg.norm(xy[wr] - xy[sh]) / scale)
            lz[el] += depth_scale * 0.08 * max(0.0, reach - 0.6)
            lz[wr] += depth_scale * 0.12 * max(0.0, reach - 0.6)

    joints[:, 0] = lx
    joints[:, 1] = ly
    joints[:, 2] = lz
    joints[~valid] = 0.0
    return joints, conf


def stabilize_bone_lengths(joints_seq: np.ndarray, conf_seq: np.ndarray) -> np.ndarray:
    """Keep bone segment lengths near each player's temporal median."""
    out = joints_seq.copy()
    t_count = len(out)
    # collect median lengths
    med_len: dict[tuple[int, int], float] = {}
    for a, b in SKELETON_EDGES:
        lengths = []
        for t in range(t_count):
            if conf_seq[t, a] >= CONF_THRESH and conf_seq[t, b] >= CONF_THRESH:
                lengths.append(float(np.linalg.norm(out[t, a] - out[t, b])))
        if lengths:
            med_len[(a, b)] = float(np.median(lengths))

    for t in range(t_count):
        # root from hips if possible
        if conf_seq[t, 11] >= CONF_THRESH and conf_seq[t, 12] >= CONF_THRESH:
            root_idx = None  # use breadth-first from pelvis midpoint conceptually
        # iteratively nudge child joints toward median bone length
        for a, b in SKELETON_EDGES:
            target = med_len.get((a, b))
            if target is None:
                continue
            if conf_seq[t, a] < CONF_THRESH or conf_seq[t, b] < CONF_THRESH:
                continue
            vec = out[t, b] - out[t, a]
            cur = float(np.linalg.norm(vec))
            if cur < 1e-6:
                continue
            # soft correction: blend toward median length
            scale = 0.65 * (target / cur) + 0.35
            out[t, b] = out[t, a] + vec * scale
    return out


def build_player_sequences(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_player[_pid(s)].append(s)

    sequences: dict[str, dict[str, Any]] = {}
    for pid, rows in by_player.items():
        rows = sorted(rows, key=_frame_idx)
        frames = np.asarray([_frame_idx(r) for r in rows], dtype=int)
        times = np.asarray([_time_s(r) for r in rows], dtype=float)
        local = []
        confs = []
        roots = []
        for r in rows:
            kps = _keypoints(r)
            j, c = pose2d_to_local3d(kps)
            local.append(j)
            confs.append(c)
            roots.append(estimate_root_xy(kps))
        joints = np.asarray(local, dtype=float)  # [T,17,3]
        conf_arr = np.asarray(confs, dtype=float)
        roots_arr = np.asarray(roots, dtype=float)

        # temporal smooth local joints + root
        for j in range(17):
            mask = conf_arr[:, j] >= CONF_THRESH
            if mask.sum() >= 3:
                joints[mask, j] = smooth_series(joints[mask, j], window=5)
        roots_arr = smooth_series(roots_arr, window=7)
        joints = stabilize_bone_lengths(joints, conf_arr)

        sequences[pid] = {
            "frames": frames,
            "times": times,
            "joints_local": joints,
            "conf": conf_arr,
            "roots_px": roots_arr,
        }
    return sequences


def place_on_ground(
    sequences: dict[str, dict[str, Any]],
    body_scale: float = 1.7,
) -> list[dict[str, Any]]:
    """Map local skeletons onto a shared ground plane.

    World axes for JSON/export:
      x: lateral court-ish
      y: depth along court
      z: up
    """
    # normalize roots across all players into a small ground region
    all_roots = np.vstack([seq["roots_px"] for seq in sequences.values()]) if sequences else np.zeros((1, 2))
    root_min = all_roots.min(axis=0)
    root_max = all_roots.max(axis=0)
    span = np.maximum(root_max - root_min, 1.0)

    records: list[dict[str, Any]] = []
    player_order = sorted(sequences.keys())
    lane = {pid: i for i, pid in enumerate(player_order)}

    for pid, seq in sequences.items():
        for i in range(len(seq["frames"])):
            rp = seq["roots_px"][i]
            # map image foot point into ground plane meters-ish
            gx = float(((rp[0] - root_min[0]) / span[0]) * 8.0 + 1.0)
            gy = float(((rp[1] - root_min[1]) / span[1]) * 6.0 + 1.0)
            # slight lane separation so A/B don't perfectly overlap
            gx += (lane[pid] - 0.5) * 0.35

            local = seq["joints_local"][i]
            conf = seq["conf"][i]
            # local: x lateral, y up, z depth-heuristic → world
            world = np.zeros((17, 3), dtype=float)
            world[:, 0] = gx + local[:, 0] * (body_scale * 0.45)
            world[:, 1] = gy + local[:, 2] * (body_scale * 0.45)
            # put feet near z=0
            z_body = local[:, 1] * (body_scale * 0.45)
            if np.any(conf >= CONF_THRESH):
                foot_z = []
                for idx in (15, 16):
                    if conf[idx] >= CONF_THRESH:
                        foot_z.append(z_body[idx])
                z0 = float(np.mean(foot_z)) if foot_z else float(np.min(z_body[conf >= CONF_THRESH]))
            else:
                z0 = 0.0
            world[:, 2] = z_body - z0
            # invalid joints: keep NaN for renderer skip, but JSON uses zeros + we store conf separately via validity
            for j in range(17):
                if conf[j] < CONF_THRESH:
                    world[j] = np.nan

            records.append(
                {
                    "frame_idx": int(seq["frames"][i]),
                    "time_s": float(seq["times"][i]),
                    "stable_player_id": pid,
                    "joints_3d": [
                        [None, None, None]
                        if not np.isfinite(world[j]).all()
                        else [float(world[j, 0]), float(world[j, 1]), float(world[j, 2])]
                        for j in range(17)
                    ],
                    "_joints_np": world,
                }
            )
    records.sort(key=lambda r: (r["frame_idx"], r["stable_player_id"]))
    return records


def write_skeleton_json(records: list[dict[str, Any]], path: Path) -> None:
    payload = []
    for r in records:
        payload.append(
            {
                "frame_idx": r["frame_idx"],
                "time_s": r["time_s"],
                "stable_player_id": r["stable_player_id"],
                "joints_3d": [
                    [0.0, 0.0, 0.0] if j[0] is None else j
                    for j in r["joints_3d"]
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_video(records: list[dict[str, Any]], output_mp4: Path, fps: float) -> None:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_frame[r["frame_idx"]].append(r)
    if not by_frame:
        raise ValueError("No frames to render")

    frames = sorted(by_frame)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # world bounds
    pts = []
    for r in records:
        j = r["_joints_np"]
        valid = np.isfinite(j).all(axis=1)
        if valid.any():
            pts.append(j[valid])
    if pts:
        cloud = np.vstack(pts)
        xmin, ymin = cloud[:, 0].min() - 1.0, cloud[:, 1].min() - 1.0
        xmax, ymax = cloud[:, 0].max() + 1.0, cloud[:, 1].max() + 1.0
        zmax = max(2.5, float(np.nanmax(cloud[:, 2])) + 0.5)
    else:
        xmin, xmax, ymin, ymax, zmax = 0, 10, 0, 8, 3

    color_i = 0
    color_map = dict(PLAYER_COLORS)

    for fidx in range(frames[0], frames[-1] + 1):
        people = by_frame.get(fidx, [])
        fig = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor="#0B0D10")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0B0D10")

        # ground plane
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            [0, 0, 0, 0, 0],
            color="#4A5160",
            lw=1.2,
        )
        # light grid
        for x in np.linspace(xmin, xmax, 5):
            ax.plot([x, x], [ymin, ymax], [0, 0], color="#2A303A", lw=0.5)
        for y in np.linspace(ymin, ymax, 5):
            ax.plot([xmin, xmax], [y, y], [0, 0], color="#2A303A", lw=0.5)

        for person in people:
            pid = person["stable_player_id"]
            if pid not in color_map:
                color_map[pid] = FALLBACK_COLORS[color_i % len(FALLBACK_COLORS)]
                color_i += 1
            c = color_map[pid]
            joints = person["_joints_np"]
            for a, b in SKELETON_EDGES:
                if np.isfinite(joints[a]).all() and np.isfinite(joints[b]).all():
                    ax.plot(
                        [joints[a, 0], joints[b, 0]],
                        [joints[a, 1], joints[b, 1]],
                        [joints[a, 2], joints[b, 2]],
                        color=c,
                        lw=3.0,
                    )
            valid = np.isfinite(joints).all(axis=1)
            if valid.any():
                ax.scatter(
                    joints[valid, 0],
                    joints[valid, 1],
                    joints[valid, 2],
                    color=c,
                    s=18,
                    depthshade=False,
                )
                # label near head / first valid
                head = joints[0] if np.isfinite(joints[0]).all() else joints[valid][0]
                ax.text(head[0], head[1], head[2] + 0.15, pid, color=c, fontsize=11)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0, zmax)
        ax.view_init(elev=18, azim=-70)
        try:
            ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax))
        except Exception:
            pass
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title("LYKON  •  Simple 3D Skeleton Replay", color="white", fontsize=14, pad=10)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height))
        writer.write(bgr)
        plt.close(fig)

    writer.release()


def infer_fps(samples: list[dict[str, Any]]) -> float:
    times = sorted({_time_s(s) for s in samples})
    if len(times) >= 2:
        dts = np.diff(times)
        dts = dts[dts > 1e-6]
        if len(dts):
            med = float(np.median(dts))
            if med > 0:
                return min(60.0, max(10.0, 1.0 / med))
    return 30.0


def main() -> None:
    p = argparse.ArgumentParser(description="Render simple pseudo-3D skeleton replay from tracks_pose.json")
    p.add_argument("--pose-json", required=True, help="Existing tracks_pose.json (do not re-run YOLO)")
    p.add_argument("--output-dir", default="data/output/1v1_3d")
    p.add_argument("--fps", type=float, default=None, help="Override FPS (default: infer from pose JSON)")
    p.add_argument("--depth-scale", type=float, default=0.35)
    args = p.parse_args()

    pose_path = Path(args.pose_json)
    samples = json.loads(pose_path.read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("tracks_pose") or []
    if not samples:
        raise SystemExit(f"No pose samples in {pose_path}")

    sequences = build_player_sequences(samples)
    records = place_on_ground(sequences)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "skeleton_3d.json"
    mp4_path = out_dir / "skeleton_3d.mp4"
    write_skeleton_json(records, json_path)

    fps = float(args.fps) if args.fps else infer_fps(samples)
    render_video(records, mp4_path, fps=fps)

    print(
        json.dumps(
            {
                "pose_json": str(pose_path),
                "players": sorted(sequences.keys()),
                "frames": len({r["frame_idx"] for r in records}),
                "fps": fps,
                "outputs": {"skeleton_3d_json": str(json_path), "skeleton_3d_mp4": str(mp4_path)},
                "note": "Heuristic pseudo-3D visualization only. Not WHAM/SMPL metric reconstruction.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
