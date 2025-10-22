"""
Enhanced MOT-3D for multi-player tracking (12 players + 2 referees).
Improvements:
 - More permissive track birth & TTL (handles intermittent visibility)
 - Grouped label matching (player/referee family match)
 - Reactivation of recently lost tracks (reduces ID switches)
 - Otherwise same as mot3d_from_csv.py pipeline
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

# -----------------------------
# Parameters
# -----------------------------
FPS = 25.0
CSV_INPUT = "output/tracks_3d.csv"
OUT_JSON = "output/tracking3d_output_multi.json"

@dataclass
class MOT3DParams:
    fps: float = 25.0
    chi2_alpha: float = 0.997
    gating_dof: int = 3
    max_speed_m_s: float = 15.0
    z_soft_clamp: float = 0.25

    # Lifecycle tuned for partial visibility
    birth_M: int = 1
    birth_N: int = 6
    ttl_misses: int = 25
    reactivation_window: int = 25  # frames
    reactivation_dist: float = 5.0  # meters

    w_maha: float = 1.0
    w_dir: float = 0.2
    w_score: float = 0.2
    w_nviews: float = 0.1

    q_pos: float = 0.10
    q_vel: float = 0.50
    q_acc: float = 1.00
    r_x: float = 0.10
    r_y: float = 0.10
    r_z: float = 0.20

    accel_clip: float = 8.0
    do_rts_smoothing: bool = True

# -----------------------------
# Utility functions (shortened)
# -----------------------------

def as_lower_columns(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def get_col(df, name, default=None):
    name = name.lower()
    if name in df.columns:
        return df[name].values
    return np.full(len(df), default)

def chi2_threshold(dof, alpha):
    return float(chi2.ppf(alpha, dof))

def heading_dir_cost(prev_v, meas_disp):
    nv = np.linalg.norm(prev_v) * np.linalg.norm(meas_disp)
    if nv < 1e-6:
        return 1.0
    cosang = np.clip(np.dot(prev_v, meas_disp) / nv, -1.0, 1.0)
    return 1.0 - cosang

# -----------------------------
# Kalman model
# -----------------------------

class CA3DKalman:
    def __init__(self, dt, params):
        self.dt = dt
        self.params = params
        self.F = np.eye(9)
        for i in range(3):
            self.F[i, i+3] = dt
            self.F[i, i+6] = 0.5 * dt**2
            self.F[i+3, i+6] = dt
        self.H = np.zeros((3,9)); self.H[0:3,0:3] = np.eye(3)
        self.Q = np.diag([params.q_pos]*3 + [params.q_vel]*3 + [params.q_acc]*3)*max(dt,1e-3)
        self.R = np.diag([params.r_x, params.r_y, params.r_z])

    def init_state(self, z):
        x = np.zeros(9); x[0:3] = z
        P = np.diag([1,1,1,10,10,10,50,50,50])
        return x, P

    def predict(self, x, P):
        F, Q = self.F, self.Q
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    def update(self, x, P, z, R_scale=1.0):
        H, Rb = self.H, self.R * R_scale
        y = z - H @ x
        S = H @ P @ H.T + Rb
        K = P @ H.T @ np.linalg.inv(S)
        x_new = x + K @ y
        P_new = (np.eye(9) - K @ H) @ P
        maha = float(y.T @ np.linalg.inv(S) @ y)
        return x_new, P_new, maha

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Detection:
    frame: int
    t: float
    z: np.ndarray
    label: str
    score: float
    n_views: float

@dataclass
class Track:
    track_id: int
    label: str
    x: np.ndarray
    P: np.ndarray
    last_frame: int
    miss_count: int = 0
    active: bool = True
    history: List[Dict] = field(default_factory=list)

# -----------------------------
# Main MOT3D
# -----------------------------

class MOT3D:
    def __init__(self, params):
        self.params = params
        self.dt = 1.0 / params.fps
        self.kf = CA3DKalman(self.dt, params)
        self.tracks: List[Track] = []
        self.next_id = 1

    def _label_match(self, a, b):
        a, b = a.lower(), b.lower()
        if ("player" in a and "player" in b) or ("referee" in a and "referee" in b):
            return True
        return False

    def _spawn(self, det: Detection) -> Track:
        x, P = self.kf.init_state(det.z)
        tr = Track(self.next_id, det.label, x, P, det.frame)
        tr.history.append({"frame": det.frame, "t": det.t, "x": det.z.tolist(), "interp": False})
        self.next_id += 1
        return tr

    def _append_state(self, tr: Track, frame: int, t: float, interp=False):
        tr.history.append({"frame": frame, "t": t, "x": tr.x[:3].tolist(), "interp": interp})

    def _reactivate(self, det: Detection):
        # Try to reactivate an inactive track nearby in space and time
        for tr in self.tracks:
            if tr.active: continue
            if not self._label_match(tr.label, det.label): continue
            frame_gap = det.frame - tr.last_frame
            if 0 < frame_gap <= self.params.reactivation_window:
                last_pos = np.array(tr.history[-1]["x"])
                dist = np.linalg.norm(det.z - last_pos)
                if dist < self.params.reactivation_dist:
                    tr.active = True
                    tr.miss_count = 0
                    tr.last_frame = det.frame
                    tr.label = det.label
                    tr.x, tr.P = self.kf.init_state(det.z)
                    tr.history.append({"frame": det.frame, "t": det.t, "x": det.z.tolist(), "interp": False})
                    return True
        return False

    def step(self, frame, dets):
        active_tracks = [t for t in self.tracks if t.active]
        if not active_tracks:
            for d in dets:
                if not self._reactivate(d):
                    self.tracks.append(self._spawn(d))
            return

        gate = chi2_threshold(self.params.gating_dof, self.params.chi2_alpha)
        M = np.full((len(active_tracks), len(dets)), 1e6)
        preds = [self.kf.predict(t.x, t.P) for t in active_tracks]

        for i, (tr, (x_pred, P_pred)) in enumerate(zip(active_tracks, preds)):
            pos_pred, vel_pred = x_pred[:3], x_pred[3:6]
            for j, d in enumerate(dets):
                if not self._label_match(tr.label, d.label): continue
                x_upd, P_upd, maha = self.kf.update(x_pred, P_pred, d.z)
                if maha > gate: continue
                dir_c = heading_dir_cost(vel_pred, d.z - pos_pred)
                cost = maha + 0.2*dir_c - 0.1*d.score
                M[i,j] = cost

        row_ind, col_ind = linear_sum_assignment(M)
        matched = set()
        used_dets = set()

        for r,c in zip(row_ind, col_ind):
            if M[r,c] >= 1e5: continue
            tr = active_tracks[r]
            d = dets[c]
            tr.x, tr.P, _ = self.kf.update(*self.kf.predict(tr.x, tr.P), d.z)
            tr.last_frame = d.frame
            tr.miss_count = 0
            tr._label = d.label
            tr.history.append({"frame": d.frame, "t": d.t, "x": d.z.tolist(), "interp": False})
            matched.add(tr.track_id)
            used_dets.add(c)

        # unmatched detections → new or reactivated
        for j, d in enumerate(dets):
            if j in used_dets: continue
            if not self._reactivate(d):
                self.tracks.append(self._spawn(d))

        # propagate missed
        for tr in active_tracks:
            if tr.track_id not in matched:
                tr.miss_count += 1
                if tr.miss_count > self.params.ttl_misses:
                    tr.active = False
                else:
                    tr.x, tr.P = self.kf.predict(tr.x, tr.P)
                    self._append_state(tr, frame, frame/self.params.fps, interp=True)

    def run(self, detections: List[Detection]):
        by_frame = {}
        for d in detections:
            by_frame.setdefault(d.frame, []).append(d)
        for f in sorted(by_frame.keys()):
            self.step(f, by_frame[f])
        return [t for t in self.tracks if len(t.history) > 3]

# -----------------------------
# I/O
# -----------------------------

def load_detections(csv_path, fps):
    df = pd.read_csv(csv_path)
    df = as_lower_columns(df)
    frames = df["frame"].astype(int).values
    xs, ys, zs = df["x"].astype(float), df["y"].astype(float), df["z"].astype(float)
    labels = get_col(df, "label", "object")
    scores = get_col(df, "score", 1.0)
    nviews = get_col(df, "n_views", 1.0)

    detections = []
    for i in range(len(df)):
        z = np.array([xs[i], ys[i], zs[i]], float)
        detections.append(Detection(frame=int(frames[i]), t=frames[i]/fps, z=z,
                                    label=str(labels[i]), score=float(scores[i]), n_views=float(nviews[i])))
    detections.sort(key=lambda d: d.frame)
    return detections

def save_json(tracks, out_path):
    data = []
    for t in tracks:
        data.append({
            "track_id": t.track_id,
            "label": t.label,
            "history": t.history
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Main
# -----------------------------

def main():

    params = MOT3DParams(fps=FPS)
    dets = load_detections(CSV_INPUT, FPS)
    mot = MOT3D(params)
    tracks = mot.run(dets)
    print(f"[MOT3D Multi] Generated {len(tracks)} multi-object tracks.")
    save_json(tracks, OUT_JSON)
    print(f"[MOT3D Multi] Saved to {OUT_JSON}")

if __name__ == "__main__":
    main()
