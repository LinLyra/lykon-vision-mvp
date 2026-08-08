from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_pseudo3d_video(pseudo3d_json: str | Path, output_mp4: str | Path, fps: float = 30.0) -> None:
    data = json.loads(Path(pseudo3d_json).read_text(encoding="utf-8"))
    frames = data["frames"]
    edges = [tuple(e) for e in data["skeleton_edges"]]
    keys = sorted((int(k) for k in frames.keys()))
    if not keys:
        raise ValueError("No pseudo3d frames to render")

    out = str(output_mp4)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = ["#00E5FF", "#FFB000", "#A3FF12", "#FF4D8D"]

    for frame_idx in range(keys[0], keys[-1] + 1):
        people = frames.get(str(frame_idx), [])
        fig = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor="#05070A")
        ax = fig.add_subplot(111, projection="3d", facecolor="#05070A")

        # Half-court-like reference plane (not a photorealistic NBA court).
        ax.plot([0, 15, 15, 0, 0], [0, 0, 14, 14, 0], [0]*5, color="#555A66", lw=1.0)
        ax.plot([7.5, 7.5], [0, 14], [0, 0], color="#333843", lw=0.7)

        for idx, person in enumerate(people):
            joints = np.asarray(person["joints_xyz_conf"], dtype=float)
            root = np.asarray(person.get("root_xy", [7.5 + idx*2, 7.0]), dtype=float)
            # place normalized skeleton in world view; z is vertical in renderer
            X = root[0] + joints[:, 0] * 0.65
            Y = root[1] + joints[:, 2] * 0.65
            Z = 1.0 + joints[:, 1] * 0.65
            c = colors[idx % len(colors)]
            for a, b in edges:
                if joints[a, 3] > 0.2 and joints[b, 3] > 0.2:
                    ax.plot([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]], color=c, lw=3)
            visible = joints[:, 3] > 0.2
            ax.scatter(X[visible], Y[visible], Z[visible], color=c, s=14)
            label = person.get("stable_player_id", person.get("track_id"))
            ax.text(root[0], root[1], 2.25, f"{label}", color="white", fontsize=9)

        ax.set_xlim(0, 15)
        ax.set_ylim(0, 14)
        ax.set_zlim(0, 3)
        ax.view_init(elev=45, azim=-65)
        ax.set_box_aspect((15, 14, 6))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title("LYKON  •  HONOR THE INSTINCT", color="white", fontsize=15, pad=12)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height))
        writer.write(bgr)
        plt.close(fig)

    writer.release()
