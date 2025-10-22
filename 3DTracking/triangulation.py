"""
Multi-view 3D triangulation (cams 2, 4, 13) — Adaptive edition
- Uses NEW intrinsics (new_K) consistent with undistorted/rectified frames.
- True DLT triangulation + robust non-linear refinement (Huber).
- Temporal assistance: optional ±1 frame matching window to increase 3-view coverage.
- Adaptive feedback: adjusts epipolar gate and per-camera weights based on live metrics.

Outputs:
- output/tracks_3d.csv (triangulated tracks with global IDs)
- Prints live metrics every ADAPT_EVERY frames.

Requirements: numpy, pandas, opencv-python, scipy
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import least_squares
from collections import defaultdict, deque, Counter
from datetime import datetime

# ===============================
# CONFIG
# ===============================
CALIB2_PATH = "../camera_data/cam_2/calib/camera_calib.json"
CALIB4_PATH = "../camera_data/cam_4/calib/camera_calib.json"
CALIB13_PATH = "../camera_data/cam_13/calib/camera_calib.json"

TRACKS2_PATH = "output/tracking_results_rect_out2.json"
TRACKS4_PATH = "output/tracking_results_rect_out4.json"
TRACKS13_PATH = "output/tracking_results_rect_out13.json"

OUTPUT_CSV = "output/tracks_3d.csv"

FPS = 25.0

# Initial (tunable) gates — will be adapted automatically
EPI_GATE_PX_INIT = 8.0     # was 6.0 → widened to grow 3-view coverage
PITCH_GATE_M_INIT = 3.0    # if pitch gating is enabled (off by default below)

USE_EPIPOLAR = True
USE_PITCH_GATE = False

# Robust refinement
HUBER_DELTA_PX = 1.5       # tighter than 2.0
SOFT_Z_SIGMA = 0.2         # meters (0 disables soft ground prior)

# Acceptance thresholds
MAX_RMS_REPROJ_PX = 5.0   # accept only high-quality triangulations

# Global tracker
ASSOC_MAX_DIST_M = 1.5
TRACK_MAX_MISSES = 10

# World scale: 1/1000 assumes calibration t in millimeters
WORLD_SCALE = 1.0 / 1000.0

# Image size of undistorted frames (width, height)
IMAGE_SIZE_W = 3840
IMAGE_SIZE_H = 2160
IMAGE_SIZE = (IMAGE_SIZE_W, IMAGE_SIZE_H)

# Per-camera weights (will be adapted)
CAM_WEIGHTS_INIT = {"2": 0.8, "4": 1.0, "13": 0.8}

# Temporal matching window (± frames)
TEMPORAL_WINDOW = 1  # 0 = off, 1 = enable ±1 frame

# Adaptive controller settings
ADAPTIVE_ON = True
ADAPT_EVERY = 50           # frames
TARGET_RMS_P90 = 8.0       # if below this, we can consider loosening the gate to get more 3-view
TARGET_3VIEW_RATIO = 0.25  # desired minimum share of 3-view among accepted measurements
EPI_GATE_MIN, EPI_GATE_MAX = 4.0, 12.0
WEIGHT_MIN, WEIGHT_MAX = 0.5, 1.3
CAM13_CAP = 1.0            # cap cam13 weight up to 1.0 if its residual is worse

# Preferred camera for tie-breaking labels
PRIMARY_CAM = "13"

# ===============================
# Utilities
# ===============================
def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def make_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3,1)])
    return K @ Rt

def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t.reshape(3,1)

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z = v.flatten()
    return np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]], dtype=float)

def relative_pose(Ri: np.ndarray, ti: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rij = Rj @ Ri.T
    tij = tj - Rj @ Ri.T @ ti
    return Rij, tij

def essential_from_relpose(Rij: np.ndarray, tij: np.ndarray) -> np.ndarray:
    return skew(tij) @ Rij

def fundamental_from_poses(Ki: np.ndarray, Ri: np.ndarray, ti: np.ndarray,
                           Kj: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> np.ndarray:
    Rij, tij = relative_pose(Ri, ti, Rj, tj)
    E = essential_from_relpose(Rij, tij)
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F / F[2,2]

def epipolar_dist(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
    x1h = np.array([x1[0], x1[1], 1.0])
    x2h = np.array([x2[0], x2[1], 1.0])
    l2 = F @ x1h
    l1 = F.T @ x2h
    d2 = abs(x2h @ l2) / np.hypot(l2[0], l2[1])
    d1 = abs(x1h @ l1) / np.hypot(l1[0], l1[1])
    return float(0.5 * (d1 + d2))

# ===============================
# Cameras
# ===============================
@dataclass
class Camera:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    C: np.ndarray

    @staticmethod
    def from_json(name: str, path: str, image_size: Tuple[int,int]=IMAGE_SIZE) -> "Camera":
        with open(path, "r") as f:
            data = json.load(f)
        K0 = np.array(data.get("K", data.get("mtx")), dtype=float)
        rvec = np.array(data.get("rvec", data.get("rvecs")), dtype=float).reshape(-1)[:3]
        tvec = np.array(data.get("tvec", data.get("tvecs")), dtype=float).reshape(-1)[:3]
        dist = np.array(data.get("dist", data.get("distCoeffs", [0,0,0,0,0])), dtype=float).reshape(-1)
        if dist.size not in (4,5,8):
            dist = np.zeros(5, dtype=float)
        R = rodrigues_to_R(rvec)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K0.astype(np.float32), dist.astype(np.float32), image_size, alpha=0)
        new_K = new_K.astype(float)
        P = make_projection(new_K, R, tvec)
        C = camera_center(R, tvec)
        return Camera(name=name, K=new_K, R=R, t=tvec, P=P, C=C)

    def backproject_ray(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uv_h = np.array([uv[0], uv[1], 1.0], dtype=float)
        d_cam = np.linalg.inv(self.K) @ uv_h
        d_world = self.R.T @ d_cam
        d_world = d_world / np.linalg.norm(d_world)
        C = self.C.flatten()
        return C, d_world

# ===============================
# Tracking I/O
# ===============================
def foot_from_bbox(b: List[float]) -> np.ndarray:
    x, y, w_or_x2, h_or_y2 = [float(v) for v in b[:4]]
    if w_or_x2 > x and h_or_y2 > y:
        w = w_or_x2 - x
        h = h_or_y2 - y
    else:
        w = w_or_x2
        h = h_or_y2
    w = max(w, 0.0); h = max(h, 0.0)
    bias = 0.02 * h  # small bias from the bottom edge
    return np.array([x + 0.5 * w, y + h - bias], dtype=float)

def det_track_id(det: Optional[Dict[str, Any]]) -> float:
    if not det:
        return float("nan")
    val = det.get("id", det.get("track_id", -1))
    try:
        return float(int(val))
    except Exception:
        try:
            return float(val)
        except Exception:
            return float("nan")

def resolve_label(det_map: Dict[str, Optional[Dict[str, Any]]], primary_cam: str = PRIMARY_CAM) -> Optional[str]:
    """Pick the most reliable label from per-camera detections."""
    labeled = []
    for name, det in det_map.items():
        if not det:
            continue
        lab = det.get("label")
        if lab:
            labeled.append((name, str(lab)))
    if not labeled:
        return None
    counts = Counter(lab for _, lab in labeled)
    most_common = counts.most_common()
    top_label, top_count = most_common[0]
    if len(most_common) == 1:
        return top_label
    second_count = most_common[1][1] if len(most_common) > 1 else 0
    if top_count >= 2 and top_count > second_count:
        return top_label
    for name, lab in labeled:
        if name == primary_cam:
            return lab
    return labeled[0][1]

def load_tracks_generic(path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)
    frame_dict: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        for d in data:
            fr = int(d.get("frame", d.get("frame_id", 0)))
            det = {"id": int(d.get("id", d.get("track_id", -1))),
                   "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                   "score": float(d.get("score", d.get("conf", 1.0))),
                   "label": d.get("label", d.get("category", d.get("class")))}
            if det["bbox"] is not None:
                frame_dict[fr].append(det)
        return frame_dict

    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            for d in data["annotations"]:
                fr = int(d.get("frame", d.get("frame_id", 0)))
                det = {"id": int(d.get("id", d.get("track_id", -1))),
                       "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                       "score": float(d.get("score", d.get("conf", 1.0))),
                       "label": d.get("label", d.get("category", d.get("class")))}
                if det["bbox"] is not None:
                    frame_dict[fr].append(det)
            return frame_dict
        ok = True
        for k,_ in data.items():
            try: int(k)
            except: ok = False; break
        if ok:
            for k, lst in data.items():
                fr = int(k)
                for d in lst:
                    det = {"id": int(d.get("id", d.get("track_id", -1))),
                           "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                           "score": float(d.get("score", d.get("conf", 1.0))),
                           "label": d.get("label", d.get("category", d.get("class")))}
                    if det["bbox"] is not None:
                        frame_dict[fr].append(det)
            return frame_dict
    raise ValueError(f"Unsupported track JSON format for {path}")

# ===============================
# Association & temporal window
# ===============================
class AssocParams:
    def __init__(self, epi_gate_px=EPI_GATE_PX_INIT, pitch_gate_m=PITCH_GATE_M_INIT, use_epipolar=True, use_pitch_gate=USE_PITCH_GATE):
        self.epi_gate_px = float(epi_gate_px)
        self.pitch_gate_m = float(pitch_gate_m)
        self.use_epipolar = bool(use_epipolar)
        self.use_pitch_gate = bool(use_pitch_gate)

def epipolar_dist_pairwise(F_ij: Optional[np.ndarray], uva: np.ndarray, uvb: np.ndarray) -> float:
    if F_ij is None:
        return 0.0
    return epipolar_dist(F_ij, uva, uvb)

def pairwise_match_by_epipolar(dets_i, dets_j, F_ij, epi_gate_px: float):
    if not dets_i or not dets_j:
        return []
    n, m = len(dets_i), len(dets_j)
    Cmat = np.full((n,m), 1e6, dtype=float)
    for a in range(n):
        uva = foot_from_bbox(dets_i[a]["bbox"])
        for b in range(m):
            uvb = foot_from_bbox(dets_j[b]["bbox"])
            ed = epipolar_dist_pairwise(F_ij, uva, uvb) if F_ij is not None else 0.0
            if ed > epi_gate_px:
                continue
            Cmat[a,b] = ed
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(Cmat)
    except Exception:
        return []
    return [(r,c) for r,c in zip(rows, cols) if Cmat[r,c] < 1e5]

def get_windowed_dets(fr, tracks, w=TEMPORAL_WINDOW):
    """Return detections list aggregated from [fr-w .. fr+w] with frame offsets stored."""
    if w <= 0:
        return [(0, det) for det in tracks.get(fr, [])]
    out = []
    for off in range(-w, w+1):
        dets = tracks.get(fr+off, [])
        for d in dets:
            out.append((off, d))
    return out

# ===============================
# Triangulation + refinement
# ===============================
def triangulate_points(cams: List[Camera], uvs: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(cams) < 2:
        return None
    A = []
    for cam, uv in zip(cams, uvs):
        x, y = uv
        P = cam.P
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    if A.shape[0] < 4:
        return None
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]

def compute_reproj_errors(cams: List[Camera], uvs: List[np.ndarray], X: np.ndarray) -> Tuple[List[float], float]:
    errs = []
    Xh = np.array([X[0], X[1], X[2], 1.0])
    for cam, uv in zip(cams, uvs):
        xp = cam.P @ Xh
        xp = xp[:2] / xp[2]
        errs.append(float(np.linalg.norm(xp - uv)))
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')
    return errs, rms

def refine_triangulated_point(X0: np.ndarray,
                              obs: List[Tuple[Camera, np.ndarray]],
                              huber_delta_px: float,
                              cam_weights: Dict[str, float],
                              soft_z_sigma: float) -> Tuple[np.ndarray, float, List[float]]:
    def residuals(p):
        Xh = np.array([p[0], p[1], p[2], 1.0])
        errs = []
        for cam, uv in obs:
            xh = cam.P @ Xh
            xp = xh[:2] / xh[2]
            e = xp - uv
            r = np.linalg.norm(e)
            w_rob = 1.0 if r <= huber_delta_px else huber_delta_px / (r + 1e-12)
            w_cam = float(cam_weights.get(cam.name, 1.0))
            errs.extend((w_cam * w_rob * e).tolist())
        if soft_z_sigma and soft_z_sigma > 0:
            errs.append(p[2] / soft_z_sigma)
        return np.array(errs, dtype=float)

    res = least_squares(residuals, X0.reshape(3,), method='lm')
    X = res.x
    per_cam, rms = compute_reproj_errors([o[0] for o in obs], [o[1] for o in obs], X)
    return X, rms, per_cam

# ===============================
# 3D Kalman constant velocity
# ===============================
class KalmanCV3D:
    def __init__(self, dt: float, q_xy: float=0.4, q_z: float=0.08, r_xy: float=0.3, r_z: float=0.15):
        self.dt = dt
        self.x = np.zeros((6,1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1e3
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt
        qx2, qy2, qz2 = q_xy**2, q_xy**2, q_z**2
        self.Q = np.diag([qx2, qy2, qz2, qx2, qy2, qz2])
        self.H = np.zeros((3,6)); self.H[0,0]=1; self.H[1,1]=1; self.H[2,2]=1
        self.R = np.diag([r_xy**2, r_xy**2, r_z**2])
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        if not self.initialized:
            self.x[:3,0] = z.flatten()
            self.initialized = True
        y = z.reshape(3,1) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def state(self):
        return self.x.copy(), self.P.copy()

# ===============================
# Global ID tracker
# ===============================
class GlobalTracker:
    def __init__(self, dt, max_dist=1.5, max_misses=10):
        self.dt = dt
        self.max_dist = max_dist
        self.max_misses = max_misses
        self.kf = {}
        self.missed = {}
        self.track_ids = []
        self.next_id = 1

    def _predict_all(self):
        preds = []
        for tid in self.track_ids:
            self.kf[tid].predict()
            xhat,_ = self.kf[tid].state()
            preds.append(xhat[:3,0])
        return np.array(preds) if preds else np.zeros((0,3))

    def _prune_missed(self):
        to_drop = [tid for tid in self.track_ids if self.missed.get(tid,0) > self.max_misses]
        for tid in to_drop:
            self.track_ids.remove(tid)
            self.kf.pop(tid, None)
            self.missed.pop(tid, None)

    def update(self, measurements: List[Tuple[np.ndarray, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
        M = len(measurements)
        preds = self._predict_all()
        T = len(self.track_ids)

        if T == 0 and M > 0:
            out = []
            for pos, meta in measurements:
                tid = self.next_id; self.next_id += 1
                self.kf[tid] = KalmanCV3D(dt=self.dt)
                self.kf[tid].predict()
                self.kf[tid].update(pos.reshape(3,))
                self.missed[tid] = 0
                self.track_ids.append(tid)
                meta2 = dict(meta); meta2["global_id"] = tid
                out.append((tid, meta2))
            return out

        if T > 0 and M > 0:
            meas = np.stack([m[0] for m in measurements], 0)
            preds_xy = preds[:, :2]; meas_xy = meas[:, :2]
            C = np.linalg.norm(preds_xy[:,None,:] - meas_xy[None,:,:], axis=2)
            C[C > self.max_dist] = 1e6
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(C)
            used_tracks, used_meas = set(), set()
            out = []
            for r,c in zip(rows, cols):
                if C[r,c] >= 1e5: continue
                tid = self.track_ids[r]
                pos, meta = measurements[c]
                self.kf[tid].update(pos.reshape(3,))
                self.missed[tid] = 0
                used_tracks.add(tid); used_meas.add(c)
                meta2 = dict(meta); meta2["global_id"] = tid
                out.append((tid, meta2))
            for m_idx in range(M):
                if m_idx in used_meas: continue
                tid = self.next_id; self.next_id += 1
                self.kf[tid] = KalmanCV3D(dt=self.dt)
                self.kf[tid].predict()
                self.kf[tid].update(measurements[m_idx][0].reshape(3,))
                self.missed[tid] = 0
                self.track_ids.append(tid)
                meta2 = dict(measurements[m_idx][1]); meta2["global_id"] = tid
                out.append((tid, meta2))
            for tid in self.track_ids:
                if tid not in used_tracks:
                    self.missed[tid] = self.missed.get(tid,0) + 1
            self._prune_missed()
            return out

        if T > 0 and M == 0:
            for tid in self.track_ids:
                self.missed[tid] = self.missed.get(tid,0) + 1
            self._prune_missed()
            return []
        if T == 0 and M == 0:
            return []

# ===============================
# Adaptive feedback
# ===============================
class AdaptiveController:
    def __init__(self, epi_gate_px: float, cam_weights: Dict[str, float]):
        self.epi_gate_px = float(epi_gate_px)
        self.cam_weights = dict(cam_weights)
        self.buffer = deque(maxlen=ADAPT_EVERY*5)

    def observe(self, n_views: int, rms: float, per_cam_res: Dict[str, float]):
        self.buffer.append((n_views, rms, per_cam_res))

    def step(self):
        if not ADAPTIVE_ON or len(self.buffer) < ADAPT_EVERY:
            return self.epi_gate_px, self.cam_weights, 0, np.nan
        buf = list(self.buffer)[-ADAPT_EVERY:]
        n3 = sum(1 for (nv,_,_) in buf if nv == 3)
        ratio3 = n3 / max(1, len(buf))
        rms_vals = [r for (_,r,_) in buf if np.isfinite(r)]
        rms_p90 = float(np.quantile(rms_vals, 0.9)) if rms_vals else np.nan
        cams = {"2": [], "4": [], "13": []}
        for _,_,res in buf:
            for key in cams.keys():
                v = res.get(key, np.nan)
                if np.isfinite(v): cams[key].append(v)
        cam_means = {k: (float(np.mean(v)) if len(v) else np.nan) for k,v in cams.items()}
        if np.isfinite(rms_p90):
            if ratio3 < TARGET_3VIEW_RATIO and rms_p90 < TARGET_RMS_P90:
                self.epi_gate_px = min(EPI_GATE_MAX, self.epi_gate_px + 0.5)
            elif rms_p90 > TARGET_RMS_P90:
                self.epi_gate_px = max(EPI_GATE_MIN, self.epi_gate_px - 0.5)
        vals = [v for v in cam_means.values() if np.isfinite(v) and v > 0]
        if len(vals) >= 2:
            inv = {k: (1.0 / cam_means[k]) if np.isfinite(cam_means[k]) and cam_means[k] > 0 else np.nan for k in cam_means}
            s = sum(v for v in inv.values() if np.isfinite(v))
            if s > 0:
                norm = {k: inv[k] / s for k in inv if np.isfinite(inv[k])}
                for k in self.cam_weights.keys():
                    if k in norm:
                        neww = 0.8*self.cam_weights[k] + 0.2*(norm[k]*3.0)
                        neww = max(WEIGHT_MIN, min(WEIGHT_MAX, neww))
                        if k == "13": neww = min(neww, CAM13_CAP)
                        self.cam_weights[k] = neww
        print(f"[ADAPT] epi_gate_px={self.epi_gate_px:.2f} ratio3={ratio3:.2f} rms_p90={rms_p90:.2f} weights={self.cam_weights}")
        return self.epi_gate_px, self.cam_weights, ratio3, rms_p90
# ===============================
# Adaptive Logger
# ===============================
class AdaptiveLogger:
    def __init__(self, log_path="output/adaptive_metrics_log.csv"):
        self.log_path = Path(log_path)
        self.entries = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, frame_idx, epi_gate, weights, ratio3, rms_p90):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": frame_idx,
            "epi_gate_px": epi_gate,
            "rms_p90_px": rms_p90,
            "ratio_3view": ratio3,
            "w_cam2": weights["2"],
            "w_cam4": weights["4"],
            "w_cam13": weights["13"],
        }
        self.entries.append(entry)

    def save(self):
        if not self.entries:
            return
        df = pd.DataFrame(self.entries)
        df.to_csv(self.log_path, index=False)
        print(f"[LOG] Saved adaptive metrics log to {self.log_path}")
# ===============================
# Pipeline
# ===============================
def run_pipeline():
    # Cameras
    cam2 = Camera.from_json("2", CALIB2_PATH, IMAGE_SIZE)
    cam4 = Camera.from_json("4", CALIB4_PATH, IMAGE_SIZE)
    cam13 = Camera.from_json("13", CALIB13_PATH, IMAGE_SIZE)

    F_24   = fundamental_from_poses(cam2.K, cam2.R, cam2.t, cam4.K,  cam4.R,  cam4.t)  if USE_EPIPOLAR else None
    F_2_13 = fundamental_from_poses(cam2.K, cam2.R, cam2.t, cam13.K, cam13.R, cam13.t) if USE_EPIPOLAR else None
    F_4_13 = fundamental_from_poses(cam4.K, cam4.R, cam4.t, cam13.K, cam13.R, cam13.t) if USE_EPIPOLAR else None

    tracks2  = load_tracks_generic(TRACKS2_PATH)
    tracks4  = load_tracks_generic(TRACKS4_PATH)
    tracks13 = load_tracks_generic(TRACKS13_PATH)

    all_frames = sorted(set(tracks2.keys()) | set(tracks4.keys()) | set(tracks13.keys()))

    # Controllers
    assoc = AssocParams(epi_gate_px=EPI_GATE_PX_INIT, pitch_gate_m=PITCH_GATE_M_INIT, use_epipolar=USE_EPIPOLAR, use_pitch_gate=USE_PITCH_GATE)
    cam_weights = dict(CAM_WEIGHTS_INIT)
    adap = AdaptiveController(epi_gate_px=assoc.epi_gate_px, cam_weights=cam_weights)
    logger = AdaptiveLogger()

    dt = 1.0 / float(FPS)
    tracker = GlobalTracker(dt=dt, max_dist=ASSOC_MAX_DIST_M, max_misses=TRACK_MAX_MISSES)

    rows = []
    output_columns = [
        "frame", "time", "global_id", "label",
        "cam2_id", "cam4_id", "cam13_id",
        "x", "y", "z",
        "vx", "vy", "vz",
        "n_views", "rms_reproj_px",
        "res_cam2_px", "res_cam4_px", "res_cam13_px"
    ]

    def uv_from_det(det): return foot_from_bbox(det["bbox"])

    def to_world(X_m: np.ndarray) -> np.ndarray:
        return np.array([float(X_m[0]) * WORLD_SCALE,
                         float(X_m[1]) * WORLD_SCALE,
                         float(X_m[2]) * WORLD_SCALE], dtype=float)

    for idx, fr in enumerate(all_frames):
        # Temporal windowed detections
        dets2 = get_windowed_dets(fr, tracks2, TEMPORAL_WINDOW)
        dets4 = get_windowed_dets(fr, tracks4, TEMPORAL_WINDOW)
        dets13 = get_windowed_dets(fr, tracks13, TEMPORAL_WINDOW)

        # Strip offsets for matching, but keep index maps
        dets2_flat = [d for (_,d) in dets2]; dets4_flat = [d for (_,d) in dets4]; dets13_flat = [d for (_,d) in dets13]

        pairs_24   = pairwise_match_by_epipolar(dets2_flat, dets4_flat, F_24, assoc.epi_gate_px)
        pairs_2_13 = pairwise_match_by_epipolar(dets2_flat, dets13_flat, F_2_13, assoc.epi_gate_px)
        pairs_4_13 = pairwise_match_by_epipolar(dets4_flat, dets13_flat, F_4_13, assoc.epi_gate_px)

        # Merge into triplets via shared cam2
        triplets = []
        map_24 = defaultdict(list)
        for i2,i4 in pairs_24:
            map_24[i2].append(i4)
        map_2_13 = defaultdict(list)
        for i2,i13 in pairs_2_13:
            map_2_13[i2].append(i13)
        for i2, lst4 in map_24.items():
            for i4 in lst4:
                for i13 in map_2_13.get(i2, []):
                    triplets.append((i2,i4,i13))

        used2, used4, used13 = set(), set(), set()
        measurements = []

        # 3-view first
        for i2, i4, i13 in triplets:
            d2, d4, d13 = dets2_flat[i2], dets4_flat[i4], dets13_flat[i13]
            uv2, uv4, uv13 = uv_from_det(d2), uv_from_det(d4), uv_from_det(d13)
            cams = [cam2, cam4, cam13]; uvs = [uv2, uv4, uv13]
            X = triangulate_points(cams, uvs)
            if X is None: continue
            X_ref, rms, per = refine_triangulated_point(X, list(zip(cams, uvs)),
                                                        HUBER_DELTA_PX, cam_weights, SOFT_Z_SIGMA)
            if np.isfinite(rms) and rms <= MAX_RMS_REPROJ_PX:
                used2.add(i2); used4.add(i4); used13.add(i13)
                # observe for adaptation
                adap.observe(3, rms, {"2": per[0], "4": per[1], "13": per[2]})
                label = resolve_label({"2": d2, "4": d4, "13": d13})
                meta = dict(frame=fr, time=fr/FPS, n_views=3, rms_reproj_px=float(rms),
                            res_cam2_px=float(per[0]), res_cam4_px=float(per[1]), res_cam13_px=float(per[2]),
                            cam2_id=det_track_id(d2), cam4_id=det_track_id(d4), cam13_id=det_track_id(d13),
                            label=label)
                measurements.append((to_world(X_ref), meta))

        # 2-view next
        def handle_pair(iA, detA, camA, iB, detB, camB, usedA: set, usedB: set):
            if iA in usedA or iB in usedB: return None
            uvA, uvB = uv_from_det(detA), uv_from_det(detB)
            cams = [camA, camB]; uvs = [uvA, uvB]
            X = triangulate_points(cams, uvs)
            if X is None: return None
            X_ref, rms, per = refine_triangulated_point(X, list(zip(cams, uvs)),
                                                        HUBER_DELTA_PX, cam_weights, SOFT_Z_SIGMA)
            if np.isfinite(rms) and rms <= MAX_RMS_REPROJ_PX:
                usedA.add(iA); usedB.add(iB)
                # match per-cam residual ordering to names
                res_map = {"2": np.nan, "4": np.nan, "13": np.nan}
                id_map  = {"2": np.nan, "4": np.nan, "13": np.nan}
                res_map[camA.name] = float(per[0]); res_map[camB.name] = float(per[1])
                id_map[camA.name]  = det_track_id(detA); id_map[camB.name] = det_track_id(detB)
                # observe for adaptation (use available cams only)
                subset = {camA.name: per[0], camB.name: per[1]}
                adap.observe(2, rms, subset)
                label = resolve_label({camA.name: detA, camB.name: detB})
                meta = dict(frame=fr, time=fr/FPS, n_views=2, rms_reproj_px=float(rms),
                            res_cam2_px=res_map["2"], res_cam4_px=res_map["4"], res_cam13_px=res_map["13"],
                            cam2_id=id_map["2"], cam4_id=id_map["4"], cam13_id=id_map["13"],
                            label=label)
                return (to_world(X_ref), meta)
            return None

        for i2,i4 in pairs_24:
            out = handle_pair(i2, dets2_flat[i2], cam2, i4, dets4_flat[i4], cam4, used2, used4)
            if out is not None: measurements.append(out)

        for i2,i13 in pairs_2_13:
            out = handle_pair(i2, dets2_flat[i2], cam2, i13, dets13_flat[i13], cam13, used2, used13)
            if out is not None: measurements.append(out)

        for i4,i13 in pairs_4_13:
            out = handle_pair(i4, dets4_flat[i4], cam4, i13, dets13_flat[i13], cam13, used4, used13)
            if out is not None: measurements.append(out)

        # Update tracker
        assigned = tracker.update(measurements)
        for tid, meta in assigned:
            xhat,_ = tracker.kf[tid].state()
            rows.append(dict(
                frame=meta["frame"], time=meta["time"], global_id=tid,
                label=meta.get("label", None),
                cam2_id=meta.get("cam2_id", np.nan),
                cam4_id=meta.get("cam4_id", np.nan),
                cam13_id=meta.get("cam13_id", np.nan),
                x=xhat[0,0], y=xhat[1,0], z=xhat[2,0],
                vx=xhat[3,0], vy=xhat[4,0], vz=xhat[5,0],
                n_views=meta["n_views"], rms_reproj_px=meta["rms_reproj_px"],
                res_cam2_px=meta["res_cam2_px"], res_cam4_px=meta["res_cam4_px"], res_cam13_px=meta["res_cam13_px"]
            ))

        # Adapt gates/weights
        if ADAPTIVE_ON and (idx+1) % ADAPT_EVERY == 0:
            new_gate, new_weights, ratio3, rms_p90 = adap.step()
            assoc.epi_gate_px = new_gate
            cam_weights.update(new_weights)
            logger.log(fr, new_gate, cam_weights, ratio3, rms_p90)

    # Save
    df = pd.DataFrame(rows, columns=output_columns)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote {len(df)} rows to {OUTPUT_CSV}")


    # Save logs and plot
    logger.save()
    try:
        log_df = pd.read_csv("output/adaptive_metrics_log.csv")
        plt.figure(figsize=(10,6))
        plt.plot(log_df["frame"], log_df["epi_gate_px"], label="Epi gate (px)", linewidth=2)
        plt.plot(log_df["frame"], log_df["w_cam2"], label="w_cam2", linestyle="--")
        plt.plot(log_df["frame"], log_df["w_cam4"], label="w_cam4", linestyle="--")
        plt.plot(log_df["frame"], log_df["w_cam13"], label="w_cam13", linestyle="--")
        plt.xlabel("Frame"); plt.ylabel("Value"); plt.title("Adaptive parameters evolution")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("output/adaptive_metrics_plot.png", dpi=150)
        print("[PLOT] Saved plot to output/adaptive_metrics_plot.png")    
    except Exception as e:
        print("[WARN] Could not plot adaptive metrics:", e)

if __name__ == "__main__":
    run_pipeline()
