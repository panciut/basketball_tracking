
#!/usr/bin/env python3
"""
Multi-view 3D player triangulation pipeline (cams 2, 4, 13)
Constants-based configuration (no CLI args) + GLOBAL ID TRACKER.
- Does NOT rely on per-camera track IDs.
- Builds persistent global IDs via 3D nearest-neighbor + Hungarian assignment with gating.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import least_squares
from collections import defaultdict

# ===============================
# CONSTANTS — EDIT THESE PATHS
# ===============================
CALIB2_PATH = "../camera_data/cam_2/calib/camera_calib.json"
CALIB4_PATH = "../camera_data/cam_4/calib/camera_calib.json"
CALIB13_PATH = "../camera_data/cam_13/calib/camera_calib.json"

TRACKS2_PATH = "output/tracking_results_rect_out2.json"
TRACKS4_PATH = "output/tracking_results_rect_out4.json"
TRACKS13_PATH = "output/tracking_results_rect_out13.json"

OUTPUT_CSV = "output/tracks_3d.csv"
# Runtime / tuning
FPS = 25.0
USE_EPIPOLAR = True
SINGLE_VIEW_FALLBACK = False
EPI_GATE_PX = 15.0
PITCH_GATE_M = 3.0
HUBER_DELTA_PX = 2.0
MAX_RMS_REPROJ_PX = 100.0

# Global tracker params
ASSOC_MAX_DIST_M = 1.5    # max allowed distance for track-measurement association
TRACK_MAX_MISSES = 10     # drop a track if not seen for this many frames

WORLD_SCALE = 1.0 / 1000.0  # convert millimeter calibration units to meters

# ===============================
# Utility math helpers
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
# Camera model
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
    def from_json(name: str, path: str) -> "Camera":
        with open(path, "r") as f:
            data = json.load(f)
        K = np.array(data.get("K", data.get("mtx")), dtype=float)
        rvec = np.array(data.get("rvec", data.get("rvecs")), dtype=float).reshape(-1)[:3]
        tvec = np.array(data.get("tvec", data.get("tvecs")), dtype=float).reshape(-1)[:3]
        R = rodrigues_to_R(rvec)
        P = make_projection(K, R, tvec)
        C = camera_center(R, tvec)
        return Camera(name=name, K=K, R=R, t=tvec, P=P, C=C)

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
    if w < 0 or h < 0:
        w = max(w, 0.0)
        h = max(h, 0.0)
    return np.array([x + 0.5 * w, y + h], dtype=float)

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

def load_tracks_generic(path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)

    frame_dict: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        for d in data:
            fr = int(d.get("frame", d.get("frame_id", 0)))
            det = {
                "id": int(d.get("id", d.get("track_id", -1))),  # not used for identity
                "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                "score": float(d.get("score", d.get("conf", 1.0)))
            }
            if det["bbox"] is not None:
                frame_dict[fr].append(det)
        return frame_dict

    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            for d in data["annotations"]:
                fr = int(d.get("frame", d.get("frame_id", 0)))
                det = {
                    "id": int(d.get("id", d.get("track_id", -1))),
                    "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                    "score": float(d.get("score", d.get("conf", 1.0)))
                }
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
                    det = {
                        "id": int(d.get("id", d.get("track_id", -1))),
                        "bbox": d.get("bbox") or d.get("tlwh") or d.get("xywh"),
                        "score": float(d.get("score", d.get("conf", 1.0)))
                    }
                    if det["bbox"] is not None:
                        frame_dict[fr].append(det)
            return frame_dict

    raise ValueError(f"Unsupported track JSON format for {path}")

# ===============================
# Association
# ===============================
class AssocParams:
    def __init__(self, epi_gate_px=3.0, pitch_gate_m=1.0, use_epipolar=True, use_pitch_gate=True):
        self.epi_gate_px = epi_gate_px
        self.pitch_gate_m = pitch_gate_m
        self.use_epipolar = use_epipolar
        self.use_pitch_gate = use_pitch_gate

def intersect_ray_with_plane_Z0(C: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    if abs(d[2]) < 1e-8:
        return None
    a = -C[2] / d[2]
    X = C + a * d
    X[2] = 0.0
    return X

def image_to_pitch_point(cam: Camera, uv: np.ndarray) -> Optional[np.ndarray]:
    C, d = cam.backproject_ray(uv)
    X = intersect_ray_with_plane_Z0(C, d)
    if X is None:
        return None
    return X[:2] * WORLD_SCALE

def epipolar_dist_pairwise(F_ij: Optional[np.ndarray], uva: np.ndarray, uvb: np.ndarray) -> float:
    if F_ij is None:
        return 0.0
    return epipolar_dist(F_ij, uva, uvb)

def pairwise_match_by_geometry(cam_i: Camera, dets_i, cam_j: Camera, dets_j, F_ij, assoc: AssocParams):
    if not dets_i or not dets_j:
        return []

    Pi = [image_to_pitch_point(cam_i, foot_from_bbox(d["bbox"])) for d in dets_i]
    Pj = [image_to_pitch_point(cam_j, foot_from_bbox(d["bbox"])) for d in dets_j]

    n, m = len(dets_i), len(dets_j)
    Cmat = np.full((n,m), 1e6, dtype=float)

    for a in range(n):
        if Pi[a] is None: continue
        for b in range(m):
            if Pj[b] is None: continue
            cost = 0.0
            if assoc.use_pitch_gate:
                dp = np.linalg.norm(Pi[a] - Pj[b])
                if dp > assoc.pitch_gate_m:
                    continue
                cost += dp
            if assoc.use_epipolar and F_ij is not None:
                uva = foot_from_bbox(dets_i[a]["bbox"])
                uvb = foot_from_bbox(dets_j[b]["bbox"])
                ed = epipolar_dist_pairwise(F_ij, uva, uvb)
                if ed > assoc.epi_gate_px: 
                    continue
                cost += 0.1 * ed
            Cmat[a,b] = cost

    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(Cmat)
    except Exception:
        return []

    pairs = []
    for r,c in zip(rows, cols):
        if Cmat[r,c] >= 1e5: continue
        pairs.append((r,c))
    return pairs

def merge_pairs_to_groups(pairs_ij, pairs_jk):
    triplets = []
    from collections import defaultdict
    map_ij = defaultdict(list)
    for i,j in pairs_ij:
        map_ij[j].append(i)
    for j,k in pairs_jk:
        for i in map_ij.get(j, []):
            triplets.append((i,j,k))
    return triplets

# ===============================
# Refinement on plane
# ===============================
def refine_on_plane(X0_xy: np.ndarray,
                    obs: List[Tuple[Camera, np.ndarray]],
                    huber_delta_px: float = 2.0) -> Tuple[np.ndarray, float, List[float]]:
    def residuals(p):
        x,y = p
        X = np.array([x,y,0.0,1.0])
        errs = []
        for cam, uv in obs:
            xh = cam.P @ X
            xp = xh[:2] / xh[2]
            e = xp - uv
            r = np.linalg.norm(e)
            w = 1.0 if r <= huber_delta_px else huber_delta_px / (r + 1e-12)
            errs.extend((w*e).tolist())
        return np.array(errs, dtype=float)

    res = least_squares(residuals, X0_xy, method='lm')
    X = np.array([res.x[0], res.x[1], 0.0])
    per_cam = []
    for cam, uv in obs:
        Xh = np.array([X[0], X[1], 0.0, 1.0])
        xp = (cam.P @ Xh); xp = xp[:2] / xp[2]
        per_cam.append(float(np.linalg.norm(xp - uv)))
    rms = float(np.sqrt(np.mean(np.square(per_cam)))) if per_cam else 1e9
    return X, rms, per_cam

# ===============================
# 3D Constant-Velocity Kalman
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
# Global ID tracker (no reliance on per-camera IDs)
# ===============================
class GlobalTracker:
    def __init__(self, dt, max_dist=1.5, max_misses=10):
        self.dt = dt
        self.max_dist = max_dist
        self.max_misses = max_misses
        self.kf = {}           # track_id -> KalmanCV3D
        self.missed = {}       # track_id -> consecutive misses
        self.track_ids = []    # active track order
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
        assigned = []
        M = len(measurements)
        preds = self._predict_all()  # (T,3)
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
            meas = np.stack([m[0] for m in measurements], 0)  # (M,3)
            preds_xy = preds[:, :2]; meas_xy = meas[:, :2]
            C = np.linalg.norm(preds_xy[:,None,:] - meas_xy[None,:,:], axis=2)  # (T,M)
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
                used_tracks.add(tid)
                used_meas.add(c)
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
# Pipeline
# ===============================
def run_pipeline():
    cam2 = Camera.from_json("2", CALIB2_PATH)
    cam4 = Camera.from_json("4", CALIB4_PATH)
    cam13 = Camera.from_json("13", CALIB13_PATH)

    F_24   = fundamental_from_poses(cam2.K, cam2.R, cam2.t, cam4.K,  cam4.R,  cam4.t)  if USE_EPIPOLAR else None
    F_2_13 = fundamental_from_poses(cam2.K, cam2.R, cam2.t, cam13.K, cam13.R, cam13.t) if USE_EPIPOLAR else None
    F_4_13 = fundamental_from_poses(cam4.K, cam4.R, cam4.t, cam13.K, cam13.R, cam13.t) if USE_EPIPOLAR else None

    tracks2  = load_tracks_generic(TRACKS2_PATH)
    tracks4  = load_tracks_generic(TRACKS4_PATH)
    tracks13 = load_tracks_generic(TRACKS13_PATH)

    all_frames = sorted(set(tracks2.keys()) | set(tracks4.keys()) | set(tracks13.keys()))
    assoc = AssocParams(epi_gate_px=EPI_GATE_PX, pitch_gate_m=PITCH_GATE_M, use_epipolar=USE_EPIPOLAR, use_pitch_gate=True)

    dt = 1.0 / float(FPS)
    tracker = GlobalTracker(dt=dt, max_dist=ASSOC_MAX_DIST_M, max_misses=TRACK_MAX_MISSES)

    rows = []
    output_columns = [
        "frame", "time", "global_id",
        "cam2_id", "cam4_id", "cam13_id",
        "x", "y", "z",
        "vx", "vy", "vz",
        "n_views", "rms_reproj_px",
        "res_cam2_px", "res_cam4_px", "res_cam13_px"
    ]

    def uv_from_det(det): return foot_from_bbox(det["bbox"])

    def to_world_measurement(X_mm: np.ndarray) -> np.ndarray:
        return np.array([float(X_mm[0]) * WORLD_SCALE,
                         float(X_mm[1]) * WORLD_SCALE,
                         0.0], dtype=float)

    for fr in all_frames:
        dets2, dets4, dets13 = tracks2.get(fr, []), tracks4.get(fr, []), tracks13.get(fr, [])

        pairs_24   = pairwise_match_by_geometry(cam2, dets2, cam4,  dets4,  F_24,   assoc)
        pairs_2_13 = pairwise_match_by_geometry(cam2, dets2, cam13, dets13, F_2_13, assoc)
        pairs_4_13 = pairwise_match_by_geometry(cam4, dets4, cam13, dets13, F_4_13, assoc)

        triplets = merge_pairs_to_groups(pairs_24, [(j,i) for i,j in pairs_2_13])
        used2, used4, used13 = set(), set(), set()

        measurements = []  # list of (pos_xyz, meta)

        # 3-view first
        for i2, i4, i13 in triplets:
            d2, d4, d13 = dets2[i2], dets4[i4], dets13[i13]
            uv2, uv4, uv13 = uv_from_det(d2), uv_from_det(d4), uv_from_det(d13)

            pts = []
            for cam, uv in [(cam2,uv2),(cam4,uv4),(cam13,uv13)]:
                C, direc = cam.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is not None: pts.append(X)
            if len(pts) < 2:
                continue
            used2.add(i2); used4.add(i4); used13.add(i13)
            X0 = np.mean(np.stack(pts,0), 0)
            X, rms, per_cam = refine_on_plane(X0[:2], [(cam2,uv2),(cam4,uv4),(cam13,uv13)], huber_delta_px=HUBER_DELTA_PX)

            if np.isfinite(rms) and rms > MAX_RMS_REPROJ_PX:
                continue
            meta = dict(
                frame=fr,
                time=fr / FPS,
                n_views=3,
                rms_reproj_px=rms,
                res_cam2_px=per_cam[0],
                res_cam4_px=per_cam[1],
                res_cam13_px=per_cam[2],
                cam2_id=det_track_id(d2),
                cam4_id=det_track_id(d4),
                cam13_id=det_track_id(d13),
            )
            measurements.append((to_world_measurement(X), meta))

        # 2-view groups
        for i2,i4 in pairs_24:
            if i2 in used2 or i4 in used4: continue
            d2, d4 = dets2[i2], dets4[i4]
            uv2, uv4 = uv_from_det(d2), uv_from_det(d4)
            pts = []
            for cam, uv in [(cam2,uv2),(cam4,uv4)]:
                C, direc = cam.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is not None: pts.append(X)
            if len(pts) < 2:
                continue
            used2.add(i2); used4.add(i4)
            X0 = np.mean(np.stack(pts,0), 0)
            X, rms, per_cam = refine_on_plane(X0[:2], [(cam2,uv2),(cam4,uv4)], huber_delta_px=HUBER_DELTA_PX)
            if np.isfinite(rms) and rms > MAX_RMS_REPROJ_PX:
                continue
            meta = dict(
                frame=fr,
                time=fr / FPS,
                n_views=2,
                rms_reproj_px=rms,
                res_cam2_px=per_cam[0],
                res_cam4_px=per_cam[1],
                res_cam13_px=np.nan,
                cam2_id=det_track_id(d2),
                cam4_id=det_track_id(d4),
                cam13_id=np.nan,
            )
            measurements.append((to_world_measurement(X), meta))

        for i2,i13 in pairs_2_13:
            if i2 in used2 or i13 in used13: continue
            d2, d13 = dets2[i2], dets13[i13]
            uv2, uv13 = uv_from_det(d2), uv_from_det(d13)
            pts = []
            for cam, uv in [(cam2,uv2),(cam13,uv13)]:
                C, direc = cam.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is not None: pts.append(X)
            if len(pts) < 2:
                continue
            used2.add(i2); used13.add(i13)
            X0 = np.mean(np.stack(pts,0), 0)
            X, rms, per_cam = refine_on_plane(X0[:2], [(cam2,uv2),(cam13,uv13)], huber_delta_px=HUBER_DELTA_PX)
            if np.isfinite(rms) and rms > MAX_RMS_REPROJ_PX:
                continue
            meta = dict(
                frame=fr,
                time=fr / FPS,
                n_views=2,
                rms_reproj_px=rms,
                res_cam2_px=per_cam[0],
                res_cam4_px=np.nan,
                res_cam13_px=per_cam[1],
                cam2_id=det_track_id(d2),
                cam4_id=np.nan,
                cam13_id=det_track_id(d13),
            )
            measurements.append((to_world_measurement(X), meta))

        for i4,i13 in pairs_4_13:
            if i4 in used4 or i13 in used13: continue
            d4, d13 = dets4[i4], dets13[i13]
            uv4, uv13 = uv_from_det(d4), uv_from_det(d13)
            pts = []
            for cam, uv in [(cam4,uv4),(cam13,uv13)]:
                C, direc = cam.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is not None: pts.append(X)
            if len(pts) < 2:
                continue
            used4.add(i4); used13.add(i13)
            X0 = np.mean(np.stack(pts,0), 0)
            X, rms, per_cam = refine_on_plane(X0[:2], [(cam4,uv4),(cam13,uv13)], huber_delta_px=HUBER_DELTA_PX)
            if np.isfinite(rms) and rms > MAX_RMS_REPROJ_PX:
                continue
            meta = dict(
                frame=fr,
                time=fr / FPS,
                n_views=2,
                rms_reproj_px=rms,
                res_cam2_px=np.nan,
                res_cam4_px=per_cam[0],
                res_cam13_px=per_cam[1],
                cam2_id=np.nan,
                cam4_id=det_track_id(d4),
                cam13_id=det_track_id(d13),
            )
            measurements.append((to_world_measurement(X), meta))

        # Single-view fallbacks (optional)
        if SINGLE_VIEW_FALLBACK:
            for idx, d in enumerate(dets2):
                if idx in used2: continue
                uv = uv_from_det(d); C, direc = cam2.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is None: continue
                meta = dict(
                    frame=fr,
                    time=fr / FPS,
                    n_views=1,
                    rms_reproj_px=np.nan,
                    res_cam2_px=np.nan,
                    res_cam4_px=np.nan,
                    res_cam13_px=np.nan,
                    cam2_id=det_track_id(d),
                    cam4_id=np.nan,
                    cam13_id=np.nan,
                )
                measurements.append((to_world_measurement(X), meta))
            for idx, d in enumerate(dets4):
                if idx in used4: continue
                uv = uv_from_det(d); C, direc = cam4.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is None: continue
                meta = dict(
                    frame=fr,
                    time=fr / FPS,
                    n_views=1,
                    rms_reproj_px=np.nan,
                    res_cam2_px=np.nan,
                    res_cam4_px=np.nan,
                    res_cam13_px=np.nan,
                    cam2_id=np.nan,
                    cam4_id=det_track_id(d),
                    cam13_id=np.nan,
                )
                measurements.append((to_world_measurement(X), meta))
            for idx, d in enumerate(dets13):
                if idx in used13: continue
                uv = uv_from_det(d); C, direc = cam13.backproject_ray(uv)
                X = intersect_ray_with_plane_Z0(C, direc)
                if X is None: continue
                meta = dict(
                    frame=fr,
                    time=fr / FPS,
                    n_views=1,
                    rms_reproj_px=np.nan,
                    res_cam2_px=np.nan,
                    res_cam4_px=np.nan,
                    res_cam13_px=np.nan,
                    cam2_id=np.nan,
                    cam4_id=np.nan,
                    cam13_id=det_track_id(d),
                )
                measurements.append((to_world_measurement(X), meta))

        # Update global tracker and record output
        assigned = tracker.update(measurements)
        for tid, meta in assigned:
            xhat,_ = tracker.kf[tid].state()
            rows.append(dict(
                frame=meta["frame"], time=meta["time"], global_id=tid,
                cam2_id=meta.get("cam2_id", np.nan),
                cam4_id=meta.get("cam4_id", np.nan),
                cam13_id=meta.get("cam13_id", np.nan),
                x=xhat[0,0], y=xhat[1,0], z=xhat[2,0],
                vx=xhat[3,0], vy=xhat[4,0], vz=xhat[5,0],
                n_views=meta["n_views"], rms_reproj_px=meta["rms_reproj_px"],
                res_cam2_px=meta["res_cam2_px"], res_cam4_px=meta["res_cam4_px"], res_cam13_px=meta["res_cam13_px"]
            ))

    if rows:
        df = pd.DataFrame(rows)
        df = df.loc[:, output_columns].sort_values(["frame","global_id"])
    else:
        df = pd.DataFrame(columns=output_columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()
