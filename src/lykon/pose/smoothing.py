"""Lightweight temporal pose smoothing for basketball motion.

Preserves fast actions (shot / crossover) while reducing jitter.
Default: Savitzky–Golay with a short window. One Euro available as alternative.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np

from lykon.schema import coerce_sample, normalize_keypoints

SmoothMethod = Literal["savgol", "one_euro", "none"]


class OneEuroFilter1D:
    """1D One Euro filter (Casiez et al.)."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = 0.0
            return x
        dt = max(1e-6, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return float(x_hat)


def _savgol_1d(y: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    if len(y) < window or window < 3:
        return y
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y
    polyorder = min(polyorder, window - 1)
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")
    except Exception:
        # fallback: light moving average
        k = np.ones(window) / window
        pad = window // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(yp, k, mode="valid")


def smooth_keypoints_sequence(
    keypoints_list: list[list[list[float]]],
    times: list[float],
    *,
    method: SmoothMethod = "savgol",
    window: int = 5,
    conf_thresh: float = 0.15,
) -> list[list[list[float]]]:
    """Smooth a temporal sequence of 17 COCO keypoints [x,y,c]."""
    if not keypoints_list:
        return []
    arr = np.asarray([normalize_keypoints(k) for k in keypoints_list], dtype=float)  # [T,17,3]
    t = len(arr)
    out = arr.copy()

    if method == "none" or t < 3:
        return out.tolist()

    if method == "one_euro":
        for j in range(17):
            fx = OneEuroFilter1D()
            fy = OneEuroFilter1D()
            for i in range(t):
                c = float(arr[i, j, 2])
                if c < conf_thresh:
                    continue
                out[i, j, 0] = fx(float(arr[i, j, 0]), float(times[i]))
                out[i, j, 1] = fy(float(arr[i, j, 1]), float(times[i]))
                # keep original confidence
                out[i, j, 2] = c
        return out.tolist()

    # savgol (default): short window to keep basketball dynamics
    for j in range(17):
        conf = arr[:, j, 2]
        mask = conf >= conf_thresh
        if mask.sum() < 3:
            continue
        idx = np.where(mask)[0]
        for dim in (0, 1):
            y = arr[idx, j, dim]
            ys = _savgol_1d(y, window=window, polyorder=2)
            out[idx, j, dim] = ys
        out[:, j, 2] = conf
    return out.tolist()


def apply_pose_smoothing(
    samples: list[dict[str, Any]],
    *,
    method: SmoothMethod = "savgol",
    window: int = 5,
) -> list[dict[str, Any]]:
    """Attach keypoints_raw / keypoints_smoothed; set keypoints to smoothed."""
    if not samples:
        return []

    normalized = [coerce_sample(s) for s in samples]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in normalized:
        by_player[str(s["stable_player_id"])].append(s)

    smoothed_rows: list[dict[str, Any]] = []
    for pid, rows in by_player.items():
        rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
        raw_kps = [r["keypoints"] for r in rows]
        times = [float(r["time_s"]) for r in rows]
        sm_kps = smooth_keypoints_sequence(raw_kps, times, method=method, window=window)
        for r, raw, sm in zip(rows, raw_kps, sm_kps):
            item = dict(r)
            item["keypoints_raw"] = normalize_keypoints(raw)
            item["keypoints_smoothed"] = normalize_keypoints(sm)
            item["keypoints"] = normalize_keypoints(sm)
            smoothed_rows.append(item)

    smoothed_rows.sort(key=lambda r: (int(r["frame_idx"]), str(r["stable_player_id"])))
    return smoothed_rows
