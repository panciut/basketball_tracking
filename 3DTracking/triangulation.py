# triangulation_position.py — upgraded for covariance, N-view clustering
# Changes inspired by triangulation.py: full 3x3 cov export, camera-agnostic clustering, pose re-estimation.
# NEW: label-aware UV extraction: ball=center, player/referee=footpoint (used in association & triangulation)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import least_squares
from collections import defaultdict, deque, Counter
from datetime import datetime
from itertools import combinations, product

# ===============================
# CONFIG
# ===============================
CALIB2_PATH = "../camera_data/cam_2/calib/cam_2_calib_rectified.json"
CALIB4_PATH = "../camera_data/cam_4/calib/cam_4_calib_rectified.json"
CALIB13_PATH = "../camera_data/cam_13/calib/cam_13_calib_rectified.json"

TRACKS2_PATH = "output/tracking_results_rect_out2.json"
TRACKS4_PATH = "output/tracking_results_rect_out4.json"
TRACKS13_PATH = "output/tracking_results_rect_out13.json"

OUTPUT_CSV = "output/triangulations_3d.csv"  # now includes covariance columns

# Optional: camera pose refinement via field correspondences
# JSON format (example):
# {
#   "2": [{"world":[X,Y,Z], "image":[u,v]}, ...],
#   "4": [...],
#   "13":[...]
# }

FPS = 25.0

# Initial (tunable) gates — may be adapted automatically
EPI_GATE_PX_INIT = 6.0
PITCH_GATE_M_INIT = 1.5

USE_EPIPOLAR = True
USE_PITCH_GATE = True

# Robust refinement
HUBER_DELTA_PX = 1.2
SOFT_Z_SIGMA = 0.0  # meters (0 disables soft ground prior)

# Acceptance thresholds
MAX_RMS_REPROJ_PX = 7.0  # strict, but covariance-aware tracker will downweight tough points

# Global tracker (simple CV-3D for smoothing IDs across frames)
ASSOC_MAX_DIST_M = 1.5
TRACK_MAX_MISSES = 5

# World scale: 1/1000 assumes calibration t in millimeters
WORLD_SCALE = 1.0 / 1000.0

# Image size of undistorted frames (width, height)
IMAGE_SIZE_W = 3840
IMAGE_SIZE_H = 2160
IMAGE_SIZE = (IMAGE_SIZE_W, IMAGE_SIZE_H)

# Per-camera weights (will be adapted)
CAM_WEIGHTS_INIT = {"2": 1.0, "4": 0.8, "13": 0.6}

# Temporal matching window (± frames)
TEMPORAL_WINDOW = 0  # 0 = off, 1 = enable ±1 frame

# Adaptive controller settings
ADAPTIVE = False
ADAPT_EVERY = 50
TARGET_RMS_P90 = 8.0
TARGET_3VIEW_RATIO = 0.25
EPI_GATE_MIN, EPI_GATE_MAX = 4.0, 12.0
WEIGHT_MIN, WEIGHT_MAX = 0.5, 1.3
CAM13_CAP = 1.0

# Preferred camera for tie-breaking labels
PRIMARY_CAM = "13"

# Covariance regularization (in world-units squared)
COV_EIG_MIN = 1e-6   # m^2
COV_EIG_MAX = 1e+2   # m^2
COV_DAMP = 1e-10     # for JTJ inversion stability

# ===============================
# Utilities
# ===============================
def make_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3,1)])
    return K @ Rt

def skew(v: np.ndarray) -> np.ndarray:
    x,y,z = v.flatten()
    return np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]], dtype=float)

def relative_pose(Ri: np.ndarray, ti: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rij = Rj @ Ri.T
    tij = tj - Rj @ Ri.T @ ti
    return Rij, tij

def fundamental_from_poses(Ki: np.ndarray, Ri: np.ndarray, ti: np.ndarray,
                           Kj: np.ndarray, Rj: np.ndarray, tj: np.ndarray) -> np.ndarray:
    Rij, tij = relative_pose(Ri, ti, Rj, tj)
    E = skew(tij) @ Rij
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F / max(1e-12, abs(F[2,2]))

def epipolar_dist(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
    x1h = np.array([x1[0], x1[1], 1.0])
    x2h = np.array([x2[0], x2[1], 1.0])
    l2 = F @ x1h
    l1 = F.T @ x2h
    d2 = abs(x2h @ l2) / np.hypot(l2[0], l2[1])
    d1 = abs(x1h @ l1) / np.hypot(l1[0], l1[1])
    return float(0.5 * (d1 + d2))

# ===============================
# Cameras (+ optional pose refinement)
# ===============================
@dataclass
class Camera:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    C: np.ndarray
    dist: np.ndarray

    @staticmethod
    def from_json(name: str, path: str, image_size: Tuple[int,int]=IMAGE_SIZE) -> "Camera":
        with open(path, "r") as f:
            data = json.load(f)
        K = np.array(data.get("K", data.get("mtx")), dtype=float)
        rvec = np.array(data.get("rvec", data.get("rvecs")), dtype=float).reshape(-1)[:3]
        tvec = np.array(data.get("tvec", data.get("tvecs")), dtype=float).reshape(-1)[:3]
        dist = np.array(data.get("dist", data.get("distCoeffs", [0,0,0,0,0])), dtype=float).reshape(-1)
        if dist.size not in (4,5,8):
            dist = np.zeros(5, dtype=float)
        R, _ = cv2.Rodrigues(rvec.reshape(3,1))

        P = make_projection(K, R, tvec)
        C = -R.T @ tvec.reshape(3,1)
        return Camera(name=name, K=K, R=R, t=tvec, P=P, C=C, dist=dist)

    def update_pose(self, R: np.ndarray, t: np.ndarray):
        self.R = R.copy()
        self.t = t.reshape(3,)
        self.P = make_projection(self.K, self.R, self.t)
        self.C = -self.R.T @ self.t.reshape(3,1)

# ===============================
# Tracking I/O
# ===============================
def foot_from_bbox(b: List[float]) -> np.ndarray:
    x, y, w_or_x2, h_or_y2 = [float(v) for v in b[:4]]
    if w_or_x2 > x and h_or_y2 > y:
        w = w_or_x2 - x; h = h_or_y2 - y
    else:
        w = w_or_x2; h = h_or_y2
    w = max(w, 0.0); h = max(h, 0.0)
    return np.array([x + 0.5 * w, y + h], dtype=float)

def center_from_bbox(b: List[float]) -> np.ndarray:
    x, y, w_or_x2, h_or_y2 = [float(v) for v in b[:4]]
    if w_or_x2 > x and h_or_y2 > y:
        w = w_or_x2 - x; h = h_or_y2 - y
    else:
        w = w_or_x2; h = h_or_y2
    w = max(w, 0.0); h = max(h, 0.0)
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=float)

def uv_by_label(det: Dict[str, Any]) -> np.ndarray:
    """
    Label-aware image point:
      - 'ball' -> bbox center
      - 'player' or 'referee' -> footpoint
      - fallback (unknown/None) -> footpoint
    """
    lab = det.get("label", None)
    lab_s = str(lab).lower() if lab is not None else ""
    if "ball" in lab_s:
        return center_from_bbox(det["bbox"])
    if "player" in lab_s or "referee" in lab_s:
        return foot_from_bbox(det["bbox"])
    # Fallback: better to use footpoint for general on-ground actors
    return foot_from_bbox(det["bbox"])

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
# Association (pairwise epipolar) — returns matches WITH costs
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

def pairwise_match_by_epipolar_with_cost(dets_i, dets_j, F_ij, epi_gate_px: float):
    # Hungarian on epipolar distance; return list of (i_idx, j_idx, cost)
    if not dets_i or not dets_j:
        return []
    n, m = len(dets_i), len(dets_j)
    Cmat = np.full((n,m), 1e6, dtype=float)
    for a in range(n):
        uva = uv_by_label(dets_i[a])
        for b in range(m):
            uvb = uv_by_label(dets_j[b])
            ed = epipolar_dist_pairwise(F_ij, uva, uvb) if F_ij is not None else 0.0
            if ed > epi_gate_px:
                continue
            Cmat[a,b] = ed
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(Cmat)
    except Exception:
        return []
    out = []
    for r,c in zip(rows, cols):
        if Cmat[r,c] < 1e5:
            out.append((r,c,float(Cmat[r,c])))
    return out

def get_windowed_dets(fr, tracks, w=TEMPORAL_WINDOW):
    if w <= 0:
        return [(0, det) for det in tracks.get(fr, [])]
    out = []
    for off in range(-w, w+1):
        dets = tracks.get(fr+off, [])
        for d in dets:
            out.append((off, d))
    return out

# ===============================
# Triangulation + refinement (+ covariance)
# ===============================
def triangulate_points(cams: List["Camera"], uvs: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(cams) < 2:
        return None
    A = []
    for cam, uv in zip(cams, uvs):
        x, y = uv
        P = cam.P
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A, dtype=float)
    if A.shape[0] < 4:
        return None
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]

def compute_reproj_errors(cams: List["Camera"], uvs: List[np.ndarray], X: np.ndarray) -> Tuple[List[float], float]:
    errs = []
    Xh = np.array([X[0], X[1], X[2], 1.0])
    for cam, uv in zip(cams, uvs):
        xp = cam.P @ Xh
        xp = xp[:2] / xp[2]
        errs.append(float(np.linalg.norm(xp - uv)))
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')
    return errs, rms

def refine_triangulated_point_with_cov(X0: np.ndarray,
                                       obs: List[Tuple["Camera", np.ndarray]],
                                       huber_delta_px: float,
                                       cam_weights: Dict[str, float],
                                       soft_z_sigma: float) -> Tuple[np.ndarray, float, List[float], np.ndarray]:
    """Returns (X_refined, rms_px, per_cam_residual_px, COV_3x3 in *input units*, i.e., mm if t is mm)."""
    def residuals(p):
        Xh = np.array([p[0], p[1], p[2], 1.0])
        errs = []
        for cam, uv in obs:
            xh = cam.P @ Xh
            xp = xh[:2] / max(1e-12, xh[2])
            e = xp - uv
            r = np.linalg.norm(e)
            w_rob = 1.0 if r <= huber_delta_px else huber_delta_px / (r + 1e-12)
            w_cam = float(cam_weights.get(cam.name, 1.0))
            errs.extend((w_cam * w_rob * e).tolist())
        if soft_z_sigma and soft_z_sigma > 0:
            errs.append(p[2] / soft_z_sigma)  # soft prior on Z
        return np.array(errs, dtype=float)

    res = least_squares(residuals, X0.reshape(3,), method='lm', jac='2-point')
    X = res.x
    per_cam, rms = compute_reproj_errors([o[0] for o in obs], [o[1] for o in obs], X)

    # --- Covariance from Gauss-Newton: cov ≈ σ² * (Jᵀ J)⁻¹ ---
    J = res.jac  # shape: (2M [+1], 3)
    JTJ = J.T @ J
    JTJ += np.eye(3) * COV_DAMP  # numerical damping
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JTJ)

    dof = max(1, J.shape[0] - 3)
    sigma2_px = (rms ** 2)  # conservative; robust weights already downweight outliers
    cov *= sigma2_px

    # Convert to world units (meters) if current coordinates are millimetric.
    cov_m = (WORLD_SCALE ** 2) * cov

    # Eigen clamp to keep cov well-conditioned for downstream gating
    w, V = np.linalg.eigh(cov_m)
    w = np.clip(w, COV_EIG_MIN, COV_EIG_MAX)
    cov_m = (V @ np.diag(w) @ V.T)

    return X, rms, per_cam, cov_m

# ===============================
# 3D Kalman constant velocity (simple smoother for IDs)
# ===============================
class KalmanCV3D:
    def __init__(self, dt: float, q_xy: float=0.4, q_z: float=0.08, r_xy: float=0.3, r_z: float=0.10):
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
        if not ADAPTIVE or len(self.buffer) < ADAPT_EVERY:
            return self.epi_gate_px, self.cam_weights, 0, np.nan
        buf = list(self.buffer)[-ADAPT_EVERY:]
        n3 = sum(1 for (nv,_,_) in buf if nv >= 3)
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
        columns = [
            "timestamp", "frame", "epi_gate_px", "rms_p90_px",
            "ratio_3view", "w_cam2", "w_cam4", "w_cam13"
        ]
        df = pd.DataFrame(self.entries if self.entries else [], columns=columns)
        df.to_csv(self.log_path, index=False)
        if self.entries:
            print(f"[LOG] Saved adaptive metrics log to {self.log_path}")
        else:
            print(f"[LOG] No adaptive metrics collected; created empty log at {self.log_path}")

# ===============================
# N-view clustering from pairwise epipolar matches
# ===============================
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def build_clusters(dets_by_cam: Dict[str, List[Dict[str, Any]]],
                   pairwise_F: Dict[Tuple[str,str], np.ndarray],
                   assoc: AssocParams) -> List[Dict[str, List[int]]]:
    """
    Returns a list of components; each component is {cam_name: [indices]}.
    Later we'll choose at most 1 index per camera (best combo).
    """
    # Build node ids per (cam, idx)
    cams = list(dets_by_cam.keys())
    node_id = {}
    rev = {}
    offset = 0
    for c in cams:
        for i in range(len(dets_by_cam[c])):
            node_id[(c, i)] = offset
            rev[offset] = (c, i)
            offset += 1
    if offset == 0:
        return []

    uf = UnionFind(offset)
    # Build matches for all pairs
    for (ci, cj) in combinations(cams, 2):
        Fi_j = pairwise_F.get((ci, cj))
        dets_i = dets_by_cam[ci]
        dets_j = dets_by_cam[cj]
        matches = pairwise_match_by_epipolar_with_cost(dets_i, dets_j, Fi_j, assoc.epi_gate_px)
        for ii, jj, _ in matches:
            uf.union(node_id[(ci, ii)], node_id[(cj, jj)])

    # Collect components
    comp_map: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for nid in range(offset):
        root = uf.find(nid)
        cam, idx = rev[nid]
        comp_map[root][cam].append(idx)

    return list(comp_map.values())

def choose_best_combo(comp: Dict[str, List[int]],
                      dets_by_cam: Dict[str, List[Dict[str, Any]]],
                      pairwise_F: Dict[Tuple[str,str], np.ndarray],
                      max_per_cam: int = 6) -> Optional[Dict[str, int]]:
    """
    For each component, select at most one detection per camera that minimizes
    the sum of epipolar distances across all available camera pairs.
    Limits enumeration to 'max_per_cam' per camera by keeping top-scoring candidates by local degree.
    """
    cams = sorted([c for c in comp.keys() if len(comp[c]) > 0])
    if len(cams) < 2:
        return None

    # Downselect per camera (keep first max_per_cam for tractability)
    candidates = {c: comp[c][:max_per_cam] for c in cams}

    best_choice = None
    best_cost = 1e18

    # Enumerate combinations
    for choice in product(*[candidates[c] for c in cams]):
        # cost = sum of epipolar distances for all camera pairs
        total = 0.0
        valid = True
        for (a, b) in combinations(range(len(cams)), 2):
            ca, cb = cams[a], cams[b]
            ia, ib = choice[a], choice[b]
            det_a = dets_by_cam[ca][ia]; det_b = dets_by_cam[cb][ib]
            uva = uv_by_label(det_a)
            uvb = uv_by_label(det_b)
            F = pairwise_F.get((ca, cb))
            if F is None:
                continue
            d = epipolar_dist_pairwise(F, uva, uvb)
            if d > EPI_GATE_PX_INIT * 2.0:  # hard sanity cutoff
                valid = False; break
            total += d
        if not valid:
            continue
        if total < best_cost:
            best_cost = total
            best_choice = {c: idx for c, idx in zip(cams, choice)}

    return best_choice

# ===============================
# Pipeline
# ===============================
def run_pipeline():
    # Cameras
    cam2 = Camera.from_json("2", CALIB2_PATH, IMAGE_SIZE)
    cam4 = Camera.from_json("4", CALIB4_PATH, IMAGE_SIZE)
    cam13 = Camera.from_json("13", CALIB13_PATH, IMAGE_SIZE)
    cams_by_name = {"2": cam2, "4": cam4, "13": cam13}

    # Precompute fundamentals for all camera pairs (ordered tuple key)
    pairwise_F: Dict[Tuple[str,str], np.ndarray] = {}
    cam_names = sorted(list(cams_by_name.keys()))
    for (ci, cj) in combinations(cam_names, 2):
        Fi_j = fundamental_from_poses(cams_by_name[ci].K, cams_by_name[ci].R, cams_by_name[ci].t,
                                      cams_by_name[cj].K, cams_by_name[cj].R, cams_by_name[cj].t) if USE_EPIPOLAR else None
        pairwise_F[(ci, cj)] = Fi_j
        pairwise_F[(cj, ci)] = Fi_j.T if Fi_j is not None else None

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
        "frame", "time", "label",
        "cam2_id", "cam4_id", "cam13_id",
        "x", "y", "z",
        "n_views", "rms_reproj_px",
        "res_cam2_px", "res_cam4_px", "res_cam13_px",
        # new: 3x3 covariance (meters^2), symmetric packed
        "cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz", "cov_zz"
    ]

    def to_world(X_maybe_mm: np.ndarray) -> np.ndarray:
        return np.array([float(X_maybe_mm[0]) * WORLD_SCALE,
                         float(X_maybe_mm[1]) * WORLD_SCALE,
                         float(X_maybe_mm[2]) * WORLD_SCALE], dtype=float)

    for idx, fr in enumerate(all_frames):
        # Temporal windowed detections
        dets_by_cam_flat: Dict[str, List[Dict[str, Any]]] = {
            "2": [d for (_,d) in get_windowed_dets(fr, tracks2, TEMPORAL_WINDOW)],
            "4": [d for (_,d) in get_windowed_dets(fr, tracks4, TEMPORAL_WINDOW)],
            "13": [d for (_,d) in get_windowed_dets(fr, tracks13, TEMPORAL_WINDOW)],
        }

        # Build N-view clusters from all pairwise matches
        components = build_clusters(dets_by_cam_flat, pairwise_F, assoc)

        used = { "2": set(), "4": set(), "13": set() }
        measurements = []

        # First, handle >=3-view combos (prefer high-view counts)
        for comp in components:
            choice = choose_best_combo(comp, dets_by_cam_flat, pairwise_F)
            if not choice or len(choice) < 2:
                continue
            cams = []
            uvs = []
            det_map = {}
            ids_map = {"2": np.nan, "4": np.nan, "13": np.nan}
            for cname, idx_det in choice.items():
                if idx_det in used[cname]:
                    break
                cams.append(cams_by_name[cname])
                det = dets_by_cam_flat[cname][idx_det]
                uvs.append(uv_by_label(det))  # label-aware UVs
                det_map[cname] = det
                ids_map[cname] = det_track_id(det)
            else:
                # Try triangulate/refine
                X = triangulate_points(cams, uvs)
                if X is None:
                    continue
                X_ref, rms, per = None, None, None
                try:
                    X_ref, rms, per, cov_m = refine_triangulated_point_with_cov(X, list(zip(cams, uvs)),
                                                                                HUBER_DELTA_PX, cam_weights, SOFT_Z_SIGMA)
                except Exception:
                    continue
                if not (np.isfinite(rms) and rms <= MAX_RMS_REPROJ_PX):
                    continue

                # Mark used
                for cname, idx_det in choice.items():
                    used[cname].add(idx_det)

                # Observe for adaptation
                per_map = {c.name: r for c, r in zip(cams, per)}
                adap.observe(len(cams), rms, per_map)

                label = resolve_label(det_map)
                meta = dict(frame=fr, time=fr/FPS, n_views=len(cams), rms_reproj_px=float(rms),
                            res_cam2_px=float(per_map.get("2", np.nan)),
                            res_cam4_px=float(per_map.get("4", np.nan)),
                            res_cam13_px=float(per_map.get("13", np.nan)),
                            cam2_id=ids_map["2"], cam4_id=ids_map["4"], cam13_id=ids_map["13"],
                            label=label,
                            cov=cov_m)
                measurements.append((to_world(X_ref), meta))

        # Then, fall back to remaining 2-view pairs
        for (ci, cj) in combinations(cam_names, 2):
            dets_i = dets_by_cam_flat[ci]
            dets_j = dets_by_cam_flat[cj]
            matches = pairwise_match_by_epipolar_with_cost(dets_i, dets_j, pairwise_F.get((ci, cj)), assoc.epi_gate_px)
            for ii, jj, _ in matches:
                if ii in used[ci] or jj in used[cj]:
                    continue
                camA, camB = cams_by_name[ci], cams_by_name[cj]
                detA, detB = dets_i[ii], dets_j[jj]
                uvA, uvB = uv_by_label(detA), uv_by_label(detB)  # label-aware UVs
                cams = [camA, camB]; uvs = [uvA, uvB]
                X = triangulate_points(cams, uvs)
                if X is None:
                    continue
                try:
                    X_ref, rms, per, cov_m = refine_triangulated_point_with_cov(X, list(zip(cams, uvs)),
                                                                                HUBER_DELTA_PX, cam_weights, SOFT_Z_SIGMA)
                except Exception:
                    continue
                if not (np.isfinite(rms) and rms <= MAX_RMS_REPROJ_PX):
                    continue
                used[ci].add(ii); used[cj].add(jj)

                # per-camera residuals map
                res_map = {"2": np.nan, "4": np.nan, "13": np.nan}
                id_map  = {"2": np.nan, "4": np.nan, "13": np.nan}
                res_map[ci] = float(per[0]); res_map[cj] = float(per[1])
                id_map[ci]  = det_track_id(detA); id_map[cj] = det_track_id(detB)

                adap.observe(2, rms, {ci: per[0], cj: per[1]})
                label = resolve_label({ci: detA, cj: detB})
                meta = dict(frame=fr, time=fr/FPS, n_views=2, rms_reproj_px=float(rms),
                            res_cam2_px=res_map["2"], res_cam4_px=res_map["4"], res_cam13_px=res_map["13"],
                            cam2_id=id_map["2"], cam4_id=id_map["4"], cam13_id=id_map["13"],
                            label=label,
                            cov=cov_m)
                measurements.append((to_world(X_ref), meta))

        # Update tracker (smoothing + IDs)
        assigned = tracker.update(measurements)
        for tid, meta in assigned:
            xhat,_ = tracker.kf[tid].state()
            cov_m = meta.get("cov", np.eye(3) * 0.05)  # fallback small cov if missing
            # Pack symmetric covariance
            cov_xx = float(cov_m[0,0]); cov_xy = float(cov_m[0,1]); cov_xz = float(cov_m[0,2])
            cov_yy = float(cov_m[1,1]); cov_yz = float(cov_m[1,2]); cov_zz = float(cov_m[2,2])

            rows.append(dict(
                frame=meta["frame"], time=meta["time"],
                label=meta.get("label", None),
                cam2_id=meta.get("cam2_id", np.nan),
                cam4_id=meta.get("cam4_id", np.nan),
                cam13_id=meta.get("cam13_id", np.nan),
                x=xhat[0,0], y=xhat[1,0], z=xhat[2,0],
                n_views=meta["n_views"], rms_reproj_px=meta["rms_reproj_px"],
                res_cam2_px=meta["res_cam2_px"], res_cam4_px=meta["res_cam4_px"], res_cam13_px=meta["res_cam13_px"],
                cov_xx=cov_xx, cov_xy=cov_xy, cov_xz=cov_xz, cov_yy=cov_yy, cov_yz=cov_yz, cov_zz=cov_zz
            ))

        # Adapt gates/weights
        if ADAPTIVE and (idx+1) % ADAPT_EVERY == 0:
            new_gate, new_weights, ratio3, rms_p90 = adap.step()
            assoc.epi_gate_px = new_gate
            cam_weights.update(new_weights)
            logger.log(fr, new_gate, cam_weights, ratio3, rms_p90)

    # Save CSV
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=output_columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()
