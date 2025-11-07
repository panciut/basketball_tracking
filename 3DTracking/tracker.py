"""
3D Multi-Object Tracker (from triangulated CSV) -> tracking3d_output.json

Improvements applied:
- Full 3D Mahalanobis association cost (kept XY pre-gate for speed).
- Strict, view-aware reprojection gating (2v=5.5px, 3v=6.5px) + lenient ball handling.
- Optional anisotropic per-measurement covariance from CSV (cov_xx..cov_zz). Fallback: synthesized R
  with depth-axis inflation when geometry is shallow or single-view.
- Class-aware quality filter: allow single-view for the ball; clamp unrealistic |Z| for players/referees.
- Per-frame measurement dedup as before + cross-class conflict suppression (player vs referee) using
  uncertainty-aware NMS.
- Lifecycle tuning: confirm earlier for players/referees (min_hits=5). Keep gentle cost shaping
  (smoothness term, single-view penalty, switch penalty).
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as hungarian
except Exception:  # pragma: no cover
    hungarian = None

# =========================
# CONSTANTS / CONFIGURATION
# =========================

# I/O (defaults to the uploaded filename; falls back to the prior path if not found)
INPUT_CSV_CANDIDATES = [
    "triangulations_3d.csv",
    "output/triangulations_3d.csv",
]
OUTPUT_JSON = "output/tracking3d_output.json"

# Camera calibration paths (kept for project completeness; unused here)
CAM_CALIB_PATHS = {
    "cam_2":   "../camera_data/cam_2/calib/camera_calib.json",
    "cam_4":   "../camera_data/cam_4/calib/camera_calib.json",
    "cam_13":  "../camera_data/cam_13/calib/camera_calib.json",
}

# Tracking parameters
DT_DEFAULT = 1.0 / 25.0
CHI2_GATE_3DOF = 9.35           # ~97.5% (3 DoF)
CHI2_GATE_2DOF = 7.0            # ~97.5% (2 DoF)

# Lifecycle tuning (per class)
CLASS_TUNING = {
    "player":   {"min_hits_confirm": 5, "max_misses": 25, "max_speed_mps": 9.5},
    "referee":  {"min_hits_confirm": 5, "max_misses": 22, "max_speed_mps": 8.5},
    "ball":     {"min_hits_confirm": 2, "max_misses": 18, "max_speed_mps": 35.0},
    "unassigned":{"min_hits_confirm": 6, "max_misses": 18, "max_speed_mps": 9.0},
    "unknown":  {"min_hits_confirm": 6, "max_misses": 18, "max_speed_mps": 9.0},
}
MIN_HITS_CONFIRM_DEFAULT = 6
MAX_MISSES_DEFAULT = 18
MAX_SPEED_MPS_DEFAULT = 10.0

# Process noise (acceleration spectral density-like) per class (m/s^2)
ACCEL_NOISE = {
    "player":  3.0,
    "referee": 3.0,
    "ball":    30.0,
    "unassigned": 4.0,
    "unknown": 4.0,
}
BALL_GRAVITY = -9.81  # m/s^2 applied on Z for ballistic prediction

# Measurement covariance synthesis
SIGMA_POS_BASE = 0.05  # 5 cm baseline
SIGMA_POS_MIN = 0.01
SIGMA_POS_MAX = 0.6

# Depth-axis (z) inflation when we don't have anisotropic covariance
Z_INFLATION_BY_VIEWS = {1: 4.0, 2: 2.0, 3: 1.2}

# Scale from reprojection error (px) -> multiplicative factor
def sigma_scale_from_reproj(rms_px: float) -> float:
    if rms_px is None or not np.isfinite(rms_px):
        return 1.0
    rms = max(0.0, float(rms_px))
    return 1.0 + (rms / 2.0) ** 2

def sigma_scale_from_views(n_views: int) -> float:
    nv = int(n_views or 1)
    if nv >= 3: return 0.75
    if nv == 2: return 1.3
    return 1.6

def clamp_measurement_sigma(sigma: float) -> float:
    return float(np.clip(sigma, SIGMA_POS_MIN, SIGMA_POS_MAX))

# ------------------------------
# Quality filter & Dedup config
# ------------------------------

# Global default: 2 views minimum, but BALL is allowed with 1 view
MIN_VIEWS_DEFAULT = 2

# Adaptive max reprojection error thresholds by number of views (px)
REPROJ_ERR_MAX_PX_BY_VIEWS = {1: 3.0, 2: 5.5, 3: 6.5}
BALL_REPROJ_MULT = 1.5  # allow looser reprojection for ball

def _max_reproj_for_views(n_views: int, is_ball: bool=False) -> float:
    if n_views is None:
        thr = REPROJ_ERR_MAX_PX_BY_VIEWS[2]
    elif n_views >= 3:
        thr = REPROJ_ERR_MAX_PX_BY_VIEWS[3]
    else:
        thr = REPROJ_ERR_MAX_PX_BY_VIEWS.get(n_views, REPROJ_ERR_MAX_PX_BY_VIEWS[2])
    if is_ball:
        thr *= BALL_REPROJ_MULT
    return thr

# Dedup thresholds (meters) by class
DEDUP_THRESH_PLAYER_M  = 0.05
DEDUP_THRESH_REF_M     = 0.07
DEDUP_THRESH_BALL_M    = 0.12
CHI2_MERGE_3D = 5.0

# Cross-class conflict suppression
CROSS_CLASS_SUPPRESS_RADIUS = 0.12  # m

# Court / scene constraints
Z_ABS_MAX_NONBALL = 0.5  # meters (reject players/referees too far from plane)

# ===============
# Helper functions
# ===============
def cls_key(c: str) -> str:
    c = (c or "unassigned").lower()
    if "ball" in c: return "ball"
    if "ref"  in c: return "referee"
    if any(k in c for k in ("play","person","human")): return "player"
    return c

def ensure_pd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, 1e9)
    return (V * w) @ V.T

def mahalanobis_sq(z: np.ndarray, z_pred: np.ndarray, S: np.ndarray) -> float:
    try:
        y = (z - z_pred).reshape(-1, 1)
        sol = np.linalg.solve(S, y)
        return float((y.T @ sol).item())
    except np.linalg.LinAlgError:
        return float("inf")

def _trace(A):
    return float(np.trace(A))

def _clamp_covariance(R: np.ndarray) -> np.ndarray:
    R = ensure_pd(R, 1e-9)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, SIGMA_POS_MIN**2, SIGMA_POS_MAX**2)
    return ensure_pd((V * w) @ V.T, 1e-9)

def _dedup_threshold_for_class(cls: str) -> float:
    c = (cls or "unknown").lower()
    if "ball" in c: return DEDUP_THRESH_BALL_M
    if "ref"  in c: return DEDUP_THRESH_REF_M
    return DEDUP_THRESH_PLAYER_M

def _should_merge(label: str, mu: np.ndarray, Sigma: np.ndarray, z2: np.ndarray, R2: np.ndarray) -> bool:
    if float(np.linalg.norm(mu - z2)) > _dedup_threshold_for_class(label):
        return False
    d2 = mahalanobis_sq(mu, z2, Sigma + R2)
    return d2 <= CHI2_MERGE_3D

def _fuse_gaussian(points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros(3), np.eye(3)*1e3
    Prec_sum = np.zeros((3,3), dtype=float)
    rhs = np.zeros((3,1), dtype=float)
    for z, R in points:
        R = ensure_pd(R, 1e-9)
        try:
            W = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            W = np.linalg.inv(ensure_pd(R + 1e-6*np.eye(3), 1e-9))
        Prec_sum += W
        rhs += W @ z.reshape(3,1)
    Prec_sum = ensure_pd(Prec_sum, 1e-9)
    try:
        Sigma = np.linalg.inv(Prec_sum)
    except np.linalg.LinAlgError:
        Sigma = ensure_pd(np.linalg.pinv(Prec_sum), 1e-9)
    mu = (Sigma @ rhs).reshape(3)
    return mu, Sigma

def _reject_by_scene(label: str, z: np.ndarray) -> bool:
    """Return True if measurement should be rejected by simple scene priors."""
    lbl = cls_key(label)
    if lbl != "ball":
        if abs(float(z[2])) > Z_ABS_MAX_NONBALL:
            return True
    return False

def _suppress_cross_class_conflicts(meas: List[dict]) -> List[dict]:
    """Uncertainty-aware NMS across classes (player vs referee)."""
    kept = []
    meas_sorted = sorted(meas, key=lambda m: _trace(m["R"]))  # keep lower-uncertainty first
    for m in meas_sorted:
        z = m["z"]; lbl = cls_key(m["label"])
        suppressed = False
        for k in kept:
            if lbl == cls_key(k["label"]):
                continue  # same class handled by dedup already
            if np.linalg.norm(z - k["z"]) < CROSS_CLASS_SUPPRESS_RADIUS:
                # conflict: keep the earlier (lower-uncertainty) already in 'kept'
                suppressed = True
                break
        if not suppressed:
            kept.append(m)
    return kept

# ==================
# Kalman Filter (CV)
# ==================
@dataclass
class KF3D:
    x: np.ndarray          # (6,1): [x y z vx vy vz]^T
    P: np.ndarray          # (6,6)
    q: float               # accel noise power (m^2/s^4)
    last_t: float

    def predict(self, t: float, gravity: Optional[float] = None) -> float:
        dt = max(1e-6, float(t - self.last_t))
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        dt2 = dt*dt; dt3 = dt2*dt; dt4 = dt2*dt2
        q = float(self.q)
        Qpos = 0.25 * dt4 * q
        Qpv  = 0.5  * dt3 * q
        Qvel =        dt2 * q
        Q = np.array([
            [Qpos, 0,    0,    Qpv,  0,    0   ],
            [0,    Qpos, 0,    0,    Qpv,  0   ],
            [0,    0,    Qpos, 0,    0,    Qpv ],
            [Qpv,  0,    0,    Qvel, 0,    0   ],
            [0,    Qpv,  0,    0,    Qvel, 0   ],
            [0,    0,    Qpv,  0,    0,    Qvel],
        ], dtype=float)
        self.x = (F @ self.x)
        if gravity is not None:
            g = float(gravity)
            self.x[2,0] += 0.5 * g * dt2
            self.x[5,0] += g * dt
        self.P = F @ self.P @ F.T + Q
        self.last_t = float(t)
        return dt

    def update(self, z: np.ndarray, R: np.ndarray):
        H = np.zeros((3,6), dtype=float)
        H[0,0]=H[1,1]=H[2,2]=1.0
        z = z.reshape(3,1)
        S = H @ self.P @ H.T + R
        S = ensure_pd(S, 1e-6)
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - (H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T  # Joseph form

    def z_pred_and_S(self, R: np.ndarray):
        H = np.zeros((3,6), dtype=float)
        H[0,0]=H[1,1]=H[2,2]=1.0
        z_pred = (H @ self.x).reshape(3)
        S = H @ self.P @ H.T + R
        return z_pred, ensure_pd(S, 1e-6)

# ===========
# Track class
# ===========
@dataclass
class Track:
    track_id: int
    label: str
    kf: KF3D
    min_hits_confirm: int
    max_misses: int
    max_speed_mps: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: List[dict] = field(default_factory=list)
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    last_dt: float = DT_DEFAULT
    switch_penalty: float = 0.0

    def to_output(self) -> dict:
        if not self.history:
            conf = 0.0
        else:
            upd = sum(1 for h in self.history if h.get("updated", False))
            total = len(self.history)
            conf = max(0.0, min(1.0, upd / max(1, total)))
        return {
            "track_id": int(self.track_id),
            "label": self.label,
            "confidence": round(float(conf), 6),
            "history": [
                {
                    "frame": int(h["frame"]),
                    "t": float(h["t"]),
                    "x": [float(h["x"][0]), float(h["x"][1]), float(h["x"][2])],
                    "interp": bool(h.get("interp", False)),
                    "conf": float(h.get("conf", 1.0)),
                } for h in self.history
            ],
        }

# =================
# Tracker container
# =================
class Tracker3D:
    def __init__(self):
        self.next_id = 1
        self.active: Dict[str, List[Track]] = {"player":[], "referee":[], "ball":[], "unassigned":[], "unknown":[]}
        self.finished: Dict[str, List[Track]] = {"player":[], "referee":[], "ball":[], "unassigned":[], "unknown":[]}

    def _new_track(self, t: float, z: np.ndarray, R: np.ndarray, label: str) -> Track:
        label = cls_key(label)
        accel = ACCEL_NOISE.get(label, ACCEL_NOISE.get("unassigned", 4.0))
        cfg = CLASS_TUNING.get(label, {})
        min_hits = int(cfg.get("min_hits_confirm", MIN_HITS_CONFIRM_DEFAULT))
        max_misses = int(cfg.get("max_misses", MAX_MISSES_DEFAULT))
        max_speed = float(cfg.get("max_speed_mps", MAX_SPEED_MPS_DEFAULT))
        kf = KF3D(
            x=np.concatenate([z.reshape(3,1), np.zeros((3,1))], axis=0),
            P=np.eye(6, dtype=float)*1.0,
            q=float(accel)**2,
            last_t=float(t)
        )
        tr = Track(
            track_id=self.next_id,
            label=label,
            kf=kf,
            min_hits_confirm=min_hits,
            max_misses=max_misses,
            max_speed_mps=max_speed,
        )
        tr.last_velocity = kf.x[3:,0].reshape(3).copy()
        tr.last_dt = DT_DEFAULT
        self.next_id += 1
        return tr

    def _predict_all(self, label: str, t: float):
        gravity = BALL_GRAVITY if label == "ball" else None
        for tr in self.active[label]:
            dt = tr.kf.predict(t, gravity=gravity)
            tr.last_dt = dt

    def _gate_and_cost(self, label: str, meas: List[Tuple[np.ndarray, np.ndarray, dict]]):
        tracks = self.active[label]
        if not tracks or not meas:
            return [], list(range(len(tracks))), list(range(len(meas)))

        C = np.full((len(tracks), len(meas)), 1e9, dtype=float)
        gating_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for i, tr in enumerate(tracks):
            vel_pred = tr.kf.x[3:,0].reshape(3)
            for j, (z, R, meta) in enumerate(meas):
                # Optional fast 2D pre-gate
                z_pred_fast, S_fast = tr.kf.z_pred_and_S(R)
                d2_xy = mahalanobis_sq(z[:2], z_pred_fast[:2], S_fast[:2, :2])
                if not np.isfinite(d2_xy) or d2_xy > CHI2_GATE_2DOF:
                    continue

                # Full 3D gating & cost
                z_pred, S = tr.kf.z_pred_and_S(R)
                d2_full = mahalanobis_sq(z, z_pred, S)
                if not np.isfinite(d2_full) or d2_full > CHI2_GATE_3DOF:
                    continue

                dt = max(1e-3, float(tr.last_dt))
                innovation = z - z_pred
                speed_xy = float(np.linalg.norm(innovation[:2]) / dt)
                max_speed = max(tr.max_speed_mps, 0.5)
                if speed_xy > max_speed:
                    if speed_xy > 1.2 * max_speed:
                        continue
                    d2_full *= (speed_xy / max_speed) ** 2  # tighten cost if near limit

                smooth_penalty = float(np.linalg.norm(vel_pred - tr.last_velocity))
                switch_pen = float(tr.switch_penalty)
                quality_pen = 0.0
                if meta:
                    n_views = int(meta.get("n_views", 1) or 1)
                    if n_views <= 1:
                        quality_pen = 0.5  # gently discourage single-view

                cost = d2_full + 0.1 * smooth_penalty + switch_pen + quality_pen
                gating_cache[(i, j)] = (z_pred, S)
                C[i, j] = cost

        # Assignment
        if hungarian is not None:
            row_ind, col_ind = hungarian(C)
            candidate_pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind) if C[i, j] < 1e6]
        else:
            used_tracks = set(); used_meas = set(); candidate_pairs = []
            flat = [(C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1])]
            for cost, i, j in sorted(flat, key=lambda x: x[0]):
                if cost >= 1e6: break
                if i in used_tracks or j in used_meas: continue
                used_tracks.add(i); used_meas.add(j); candidate_pairs.append((i, j))

        # No extra gate here; C already uses full 3D d^2
        pairs = candidate_pairs
        assigned_t = {i for i,_ in pairs}
        assigned_m = {j for _,j in pairs}
        un_t = [i for i in range(len(tracks)) if i not in assigned_t]
        un_m = [j for j in range(len(meas)) if j not in assigned_m]
        return pairs, un_t, un_m

    def step(self, frame: int, t: float, measurements: Dict[str, List[Tuple[np.ndarray, np.ndarray, dict]]]):  # noqa: E501
        for label in self.active.keys():
            self._predict_all(label, t)

        for label, meas in measurements.items():
            pairs, un_t_idx, un_m_idx = self._gate_and_cost(label, meas)
            tracks = self.active[label]

            # Update matched
            for i, j in pairs:
                tr = tracks[i]
                z, R, meta = meas[j]
                z_pred, _ = tr.kf.z_pred_and_S(R)
                innovation = z - z_pred
                tr.kf.update(z, R)
                tr.hits += 1
                tr.misses = 0
                if not tr.confirmed and tr.hits >= tr.min_hits_confirm:
                    tr.confirmed = True
                tr.last_velocity = tr.kf.x[3:,0].reshape(3).copy()
                innov_xy = float(np.linalg.norm(innovation[:2]))
                tr.switch_penalty = min(5.0, max(0.0, 0.25 * tr.switch_penalty + 0.05 * innov_xy))
                tr.history.append({"frame": frame, "t": t, "x": tr.kf.x[:3,0].tolist(),
                                   "interp": False, "conf": 1.0, "updated": True})

            # Unmatched tracks -> miss
            for i in un_t_idx:
                tr = tracks[i]
                tr.misses += 1
                tr.last_velocity = tr.kf.x[3:,0].reshape(3).copy()
                tr.switch_penalty *= 0.7
                tr.history.append({"frame": frame, "t": t, "x": tr.kf.x[:3,0].tolist(),
                                   "interp": True, "conf": max(0.0, 1.0 - 0.1*tr.misses), "updated": False})

            # Unmatched measurements -> new tracks
            for j in un_m_idx:
                z, R, meta = meas[j]
                tr = self._new_track(t, z, R, label)
                tr.switch_penalty = 0.0
                tr.history.append({"frame": frame, "t": t, "x": tr.kf.x[:3,0].tolist(),
                                   "interp": False, "conf": 1.0, "updated": True})
                self.active[label].append(tr)

        # Terminate dead tracks
        for label in list(self.active.keys()):
            stay = []
            for tr in self.active[label]:
                if tr.misses > tr.max_misses:
                    self.finished[label].append(tr)
                else:
                    stay.append(tr)
            self.active[label] = stay

    def all_tracks(self) -> List['Track']:
        out = []
        for L in [self.active, self.finished]:
            for v in L.values():
                out.extend(v)
        return out

# ==================
# Data I/O utilities
# ==================
def _try_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _read_cov_from_row(row) -> Optional[np.ndarray]:
    keys = ["cov_xx","cov_xy","cov_xz","cov_yx","cov_yy","cov_yz","cov_zx","cov_zy","cov_zz"]
    has_any = any(k in row and row[k] not in (None,"","nan","NaN") for k in keys)
    if not has_any:
        return None
    # read and symmetrize
    xx=_try_float(row.get("cov_xx")); xy=_try_float(row.get("cov_xy")); xz=_try_float(row.get("cov_xz"))
    yx=_try_float(row.get("cov_yx")); yy=_try_float(row.get("cov_yy")); yz=_try_float(row.get("cov_yz"))
    zx=_try_float(row.get("cov_zx")); zy=_try_float(row.get("cov_zy")); zz=_try_float(row.get("cov_zz"))
    # default zeros if missing
    xx=xx if xx is not None else 0.0; yy=yy if yy is not None else 0.0; zz=zz if zz is not None else 0.0
    xy=xy if xy is not None else (yx if yx is not None else 0.0)
    xz=xz if xz is not None else (zx if zx is not None else 0.0)
    yz=yz if yz is not None else (zy if zy is not None else 0.0)
    R = np.array([[xx, xy, xz],
                  [xy, yy, yz],
                  [xz, yz, zz]], dtype=float)
    return _clamp_covariance(R)

def _synth_measurement_R(label: str, n_views: int, rms_px: Optional[float]) -> np.ndarray:
    sigma = SIGMA_POS_BASE * sigma_scale_from_reproj(rms_px) * sigma_scale_from_views(n_views or 1)
    sigma = clamp_measurement_sigma(sigma)
    R = (sigma**2) * np.eye(3, dtype=float)
    # Heuristic anisotropy if no explicit covariance is provided
    nv = int(n_views or 1)
    inflate = Z_INFLATION_BY_VIEWS.get(nv, 1.6)
    if cls_key(label) != "ball":
        R[2,2] *= inflate  # depth less certain for people
    return _clamp_covariance(R)

def read_csv(path: str):
    import csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frame = int(row.get("frame"))
                t = float(row.get("time"))
                label = cls_key(row.get("label","unassigned"))
                x = float(row.get("x")); y = float(row.get("y")); z = float(row.get("z"))
                n_views = int(float(row.get("n_views"))) if row.get("n_views") not in (None, "") else None
                err_px = row.get("rms_reproj_px")
                rms_px = float(err_px) if err_px not in (None, "", "nan", "NaN") else None

                # Scene rejection (non-ball too far from plane)
                if _reject_by_scene(label, np.array([x,y,z], dtype=float)):
                    continue

                # Per-measurement covariance (if present), else synthesize
                R = _read_cov_from_row(row)
                if R is None:
                    R = _synth_measurement_R(label, n_views or 1, rms_px)

                rows.append({
                    "frame": frame,
                    "t": t,
                    "label": label,
                    "z": np.array([x, y, z], dtype=float),
                    "R": R,
                    "meta": {"n_views": int(n_views or 1), "rms_reproj_px": rms_px}
                })
            except Exception:
                continue
    # group by (frame, t)
    buckets: Dict[Tuple[int, float], List[dict]] = {}
    for d in rows:
        key = (d["frame"], d["t"])
        buckets.setdefault(key, []).append(d)
    keys = sorted(buckets.keys(), key=lambda k: (k[1], k[0]))
    return keys, buckets

# --------------------------
# Per-frame filtering/dedup
# --------------------------
def _quality_filter(meas: List[dict]) -> List[dict]:
    kept = []
    for m in meas:
        lbl = cls_key(m["label"])
        n_views = int(m["meta"].get("n_views", 1) or 1)
        rms = m["meta"].get("rms_reproj_px", None)

        # Views: allow single-view only for BALL
        min_views = 1 if lbl == "ball" else MIN_VIEWS_DEFAULT
        if n_views < min_views:
            continue

        # Reprojection threshold (looser for ball)
        thr = _max_reproj_for_views(n_views, is_ball=(lbl=="ball"))
        if rms is not None and np.isfinite(rms):
            if float(rms) > thr:
                continue

        kept.append(m)
    return kept

def _dedup_frame_measurements(meas: List[dict]) -> List[dict]:
    if not meas:
        return meas
    # group indices by class
    by_cls: Dict[str, List[int]] = {}
    for i, m in enumerate(meas):
        c = cls_key(m["label"])
        by_cls.setdefault(c, []).append(i)

    deduped: List[dict] = []
    for c, idxs in by_cls.items():
        if not idxs:
            continue
        # order by ascending trace(R)
        idxs_sorted = sorted(idxs, key=lambda i: _trace(meas[i]["R"]))
        visited = set()
        for i0 in idxs_sorted:
            if i0 in visited: continue
            cluster_idxs = [i0]; visited.add(i0)
            mu, Sigma = _fuse_gaussian([(meas[i0]["z"], meas[i0]["R"])])
            for j in idxs_sorted:
                if j in visited: continue
                z2, R2 = meas[j]["z"], meas[j]["R"]
                if _should_merge(c, mu, Sigma, z2, R2):
                    cluster_idxs.append(j); visited.add(j)
                    pts = [(meas[k]["z"], meas[k]["R"]) for k in cluster_idxs]
                    mu, Sigma = _fuse_gaussian(pts)

            # aggregate metadata for the cluster
            nviews_max = max(int(meas[k]["meta"].get("n_views",1) or 1) for k in cluster_idxs)
            reproj_vals = [meas[k]["meta"].get("rms_reproj_px", None) for k in cluster_idxs]
            reproj_vals = [v for v in reproj_vals if v is not None and np.isfinite(v)]
            reproj_mean = float(np.mean(reproj_vals)) if reproj_vals else None

            deduped.append({
                "label": c,
                "z": mu.copy(),
                "R": _clamp_covariance(Sigma.copy()),
                "meta": {"n_views": nviews_max, "rms_reproj_px": reproj_mean}
            })

    # Cross-class suppression to avoid player/referee duplicates at the same spot
    deduped = _suppress_cross_class_conflicts(deduped)
    return deduped

# ============
# Main routine
# ============
def _resolve_input_csv() -> Path:
    for p in INPUT_CSV_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    # if nothing found, return the first (will raise later)
    return Path(INPUT_CSV_CANDIDATES[0])

def main():
    in_path = _resolve_input_csv()
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found. Tried: {INPUT_CSV_CANDIDATES}")
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys, buckets = read_csv(str(in_path))

    trk = Tracker3D()
    for frame, t in keys:
        # Build per-class measurement lists
        per_cls: Dict[str, List[Tuple[np.ndarray, np.ndarray, dict]]] = {
            "player":[], "referee":[], "ball":[], "unassigned":[], "unknown":[]
        }
        frame_meas = _quality_filter(buckets[(frame, t)])
        frame_meas = _dedup_frame_measurements(frame_meas)
        for d in frame_meas:
            lbl = cls_key(d["label"])
            R = _clamp_covariance(d["R"])
            per_cls[lbl].append((d["z"], R, d["meta"]))
        trk.step(frame, t, per_cls)

    # Collect all tracks (active + finished); keep everything
    tracks = trk.all_tracks()
    payload = [tr.to_output() for tr in tracks if len(tr.history) > 0]

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(payload)} tracks.")

if __name__ == "__main__":
    main()
