# triangulate.py

import os
import json
import math
import glob
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional

# -------------------------------
# Configuration
# -------------------------------

# Map camera "outX" to camera index X used in camera_data/cam_[X]/...
CAMERA_MAP = {
    "out2": 2,
    "out4": 4,
    "out13": 13,
}

TRACKING_FILES = {
    "out2": "output/tracking_results_out2.json",
    "out4": "output/tracking_results_out4.json",
    "out13": "output/tracking_results_out13.json",
}

CALIB_TEMPLATE = "../camera_data/cam_{}/calib/camera_calib.json"

# Association/triangulation thresholds
REPROJ_THRESH_PX = 4.0
MIN_CAMS_FOR_TRI = 2
HUBER_DELTA = 3.0  # px
MAX_IRLS_ITERS = 10

# Motion filtering
GRAVITY = 9.81
DT_DEFAULT = 1.0 / 25.0  # fallback if no fps known
HUMAN_Z_PRIOR = 1.4  # meters, torso center prior
HUMAN_Z_PRIOR_STD = 0.4
PLANE_Z = 0.0

# Output
OUTPUT_CSV = "triangulated_tracks.csv"

# Classes expected
CLASSES = ["ball", "referee", "player", "unassigned"]


# -------------------------------
# Utility functions
# -------------------------------

def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec.flatten() / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

def load_calib_for_cam(cam_index: int) -> Dict[str, np.ndarray]:
    path = CALIB_TEMPLATE.format(cam_index)
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["mtx"], dtype=float)
    dist = np.array(data["dist"], dtype=float).reshape(-1)
    rvec = np.array(data["rvecs"], dtype=float).reshape(3)
    tvec = np.array(data["tvecs"], dtype=float).reshape(3)
    R = rodrigues_to_R(rvec)
    t = tvec.reshape(3, 1)
    return {"K": K, "dist": dist, "R": R, "t": t}

def undistort_points_norm(pts_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    # pts_px: (N,2)
    # Return normalized undistorted points in camera coords (x,y) with z=1
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (pts_px[:,0] - cx) / fx
    y = (pts_px[:,1] - cy) / fy
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    k3 = dist[4] if dist.shape[0] > 4 else 0.0
    # Iterative undistortion (inverse distortion)
    x_u, y_u = x.copy(), y.copy()
    for _ in range(5):
        r2 = x_u*x_u + y_u*y_u
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        x_tang = 2*p1*x_u*y_u + p2*(r2 + 2*x_u*x_u)
        y_tang = p1*(r2 + 2*y_u*y_u) + 2*p2*x_u*y_u
        x_est = (x_u - x_tang) / radial
        y_est = (y_u - y_tang) / radial
        x_u, y_u = x_est, y_est
    return np.stack([x_u, y_u], axis=1)

def bbox_to_keypoint(bbox: List[float], label: str) -> Tuple[float,float]:
    x1,y1,x2,y2 = bbox
    if label in ("player", "referee"):
        # bottom-center (footpoint)
        return (0.5*(x1+x2), y2)
    else:
        # ball/unassigned -> center
        return (0.5*(x1+x2), 0.5*(y1+y2))

def build_bearing(xy_norm: np.ndarray) -> np.ndarray:
    # from normalized pixel to unit bearing in camera frame
    v = np.array([xy_norm[0], xy_norm[1], 1.0], dtype=float)
    return v / np.linalg.norm(v)

def project_point(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    Xc = R @ Xw.reshape(3,1) + t
    if Xc[2,0] <= 1e-9:
        return np.array([np.nan, np.nan])
    x = Xc[0,0] / Xc[2,0]
    y = Xc[1,0] / Xc[2,0]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]
    return np.array([u, v])

def linear_triangulate_pair(K1,R1,t1,pt1,K2,R2,t2,pt2) -> Optional[np.ndarray]:
    # DLT from two views; pt are pixel coords
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])
    x1,y1 = pt1
    x2,y2 = pt2
    A = np.array([
        x1*P1[2,:] - P1[0,:],
        y1*P1[2,:] - P1[1,:],
        x2*P2[2,:] - P2[0,:],
        y2*P2[2,:] - P2[1,:]
    ])
    # Solve AX=0
    U,S,Vt = np.linalg.svd(A)
    Xh = Vt[-1,:]
    if abs(Xh[-1]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    return X

def huber_weight(r: float, delta: float) -> float:
    a = abs(r)
    if a <= delta:
        return 1.0
    return delta / a

def refine_point_irls(X0: np.ndarray,
                      obs: List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]],  # (K,R,t,pt_px)
                      max_iters=MAX_IRLS_ITERS,
                      huber_delta=HUBER_DELTA) -> Tuple[np.ndarray, List[float]]:
    X = X0.copy()
    for _ in range(max_iters):
        J_list = []
        r_list = []
        w_list = []
        for (K,R,t,pt) in obs:
            u,v = pt
            Xc = R @ X.reshape(3,1) + t
            Z = Xc[2,0]
            if Z <= 1e-9:
                continue
            x = Xc[0,0] / Z
            y = Xc[1,0] / Z
            u_hat = K[0,0]*x + K[0,2]
            v_hat = K[1,1]*y + K[1,2]
            du = u_hat - u
            dv = v_hat - v
            # Jacobian wrt X (chain rule)
            dxdXc = np.array([[1/Z, 0, -x/Z]])
            dydXc = np.array([[0, 1/Z, -y/Z]])
            dXc_dX = R
            du_dX = K[0,0] * (dxdXc @ dXc_dX)
            dv_dX = K[1,1] * (dydXc @ dXc_dX)
            J_list.append(du_dX.reshape(3))
            r_list.append(du)
            w_list.append(huber_weight(du, huber_delta))
            J_list.append(dv_dX.reshape(3))
            r_list.append(dv)
            w_list.append(huber_weight(dv, huber_delta))
        if len(J_list) < 4:
            break
        J = np.vstack(J_list)  # (2M,3)
        r = np.array(r_list).reshape(-1,1)
        W = np.diag(np.array(w_list))
        # Solve weighted least squares: (J^T W J) dX = - J^T W r
        H = J.T @ W @ J
        g = J.T @ W @ r
        try:
            dX = -np.linalg.solve(H, g).flatten()
        except np.linalg.LinAlgError:
            break
        X = X + dX
        if np.linalg.norm(dX) < 1e-4:
            break
    # residuals
    res = []
    for (K,R,t,pt) in obs:
        proj = project_point(K,R,t,X)
        res.append(float(np.linalg.norm(proj - pt)))
    return X, res

def pairwise_seed_from_rays(cam_obs: List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]) -> Optional[np.ndarray]:
    # Build seeds by averaging DLT over all camera pairs
    seeds = []
    for i in range(len(cam_obs)):
        for j in range(i+1, len(cam_obs)):
            K1,R1,t1,pt1 = cam_obs[i]
            K2,R2,t2,pt2 = cam_obs[j]
            X = linear_triangulate_pair(K1,R1,t1,pt1,K2,R2,t2,pt2)
            if X is not None and np.isfinite(X).all():
                seeds.append(X)
    if not seeds:
        return None
    return np.median(np.stack(seeds, axis=0), axis=0)

def reproj_errors(X: np.ndarray, obs: List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]) -> List[float]:
    errs = []
    for (K,R,t,pt) in obs:
        uv = project_point(K,R,t,X)
        errs.append(float(np.linalg.norm(uv - pt)))
    return errs

# -------------------------------
# Data loading
# -------------------------------

def load_tracking(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(file_path, "r") as f:
        data = json.load(f)
    frames = {}
    for k,v in data.items():
        try:
            frames[int(k)] = v
        except:
            continue
    return frames

def load_all_tracks() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    cams = {}
    for cam_key, fp in TRACKING_FILES.items():
        cams[cam_key] = load_tracking(fp)
    return cams

def load_all_calibs() -> Dict[str, Dict[str, np.ndarray]]:
    calibs = {}
    for cam_key, cam_idx in CAMERA_MAP.items():
        calibs[cam_key] = load_calib_for_cam(cam_idx)
    return calibs

# -------------------------------
# Association within a frame
# -------------------------------

def frame_union(frames_per_cam: Dict[str, Dict[int, List[Dict]]]) -> List[int]:
    all_frames = set()
    for cam_key, frdict in frames_per_cam.items():
        all_frames.update(frdict.keys())
    return sorted(all_frames)

def detections_to_points(dets: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    pts = []
    for d in dets:
        kp = bbox_to_keypoint(d["bbox"], d.get("label","unassigned"))
        pts.append({
            "id": d.get("id", -1),
            "label": d.get("label", "unassigned"),
            "pt": kp,
            "bbox": d["bbox"]
        })
    return pts

def build_observation_sets(frame: int,
                           tracks: Dict[str, Dict[int, List[Dict]]],
                           calibs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]]:
    # Return per-class observations: {"player": [(K,R,t,pt_px),...], ...}
    per_class = defaultdict(list)
    for cam_key in tracks.keys():
        dets = tracks[cam_key].get(frame, [])
        if not dets:
            continue
        cal = calibs[cam_key]
        K, R, t = cal["K"], cal["R"], cal["t"]
        pts = detections_to_points(dets)
        for p in pts:
            pt_px = np.array(p["pt"], dtype=float)
            label = p["label"]
            per_class[label].append((K,R,t,pt_px))
    return per_class

def group_entities_within_class(frame: int,
                                tracks: Dict[str, Dict[int, List[Dict]]],
                                calibs: Dict[str, Dict[str, np.ndarray]],
                                target_count: int) -> List[List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]]:
    # Greedy geometric grouping: create up to target_count groups per class
    # Strategy: start from all observations, iteratively form groups by picking pairs with best DLT consistency.
    # This is a simplification; can be replaced with global assignment.
    all_obs = []
    for cam_key in tracks.keys():
        dets = tracks[cam_key].get(frame, [])
        if not dets:
            continue
        cal = calibs[cam_key]
        K, R, t = cal["K"], cal["R"], cal["t"]
        for d in dets:
            label = d.get("label","unassigned")
            # group only appropriate labels outside this function
            pt_px = np.array(bbox_to_keypoint(d["bbox"], label), dtype=float)
            all_obs.append((label, cam_key, (K,R,t,pt_px)))
    # This function expects pre-filtering by label prior to call.
    obs = [o[2] for o in all_obs]
    used = [False]*len(obs)
    groups = []
    # Build groups around strong pairs
    for i in range(len(obs)):
        if used[i]:
            continue
        # seed a group with obs[i]; try to add compatible others
        base = obs[i]
        group = [base]
        used[i] = True
        for j in range(len(obs)):
            if used[j] or i==j:
                continue
            seed = linear_triangulate_pair(obs[i][0], obs[i][1], obs[i][2], obs[i][3],
                                           obs[j][0], obs[j][1], obs[j][2], obs[j][3])
            if seed is None:
                continue
            errs = reproj_errors(seed, [obs[i], obs[j]])
            if np.mean(errs) <= REPROJ_THRESH_PX:
                group.append(obs[j])
                used[j] = True
        if len(group) >= 1:
            groups.append(group)
        if len(groups) >= target_count:
            break
    return groups

# -------------------------------
# Per-frame triangulation
# -------------------------------

def triangulate_group(group_obs: List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]) -> Optional[Tuple[np.ndarray, List[int], List[float]]]:
    if len(group_obs) < MIN_CAMS_FOR_TRI:
        return None
    seed = pairwise_seed_from_rays(group_obs)
    if seed is None:
        return None
    X, res = refine_point_irls(seed, group_obs)
    # Reject outlier views
    in_obs = []
    in_idx = []
    in_res = []
    for k,(r) in enumerate(res):
        pass
    # res is aligned with group_obs one-per-camera residual norm; recompute to map
    res_full = reproj_errors(X, group_obs)
    for i,e in enumerate(res_full):
        if e <= REPROJ_THRESH_PX:
            in_obs.append(group_obs[i])
            in_idx.append(i)
            in_res.append(e)
    if len(in_obs) < MIN_CAMS_FOR_TRI:
        return None
    X_ref, res_ref = refine_point_irls(X, in_obs)
    return X_ref, in_idx, res_ref

# -------------------------------
# Temporal filtering (simple)
# -------------------------------

class ConstantVelocityKF:
    def __init__(self, dt=DT_DEFAULT, q=1.0, r=0.1):
        # state: [x y z vx vy vz]
        self.dt = dt
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt
        self.H = np.zeros((3,6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.Q = q * np.diag([dt**4/4, dt**4/4, dt**4/4, dt**2, dt**2, dt**2])
        self.R = r * np.eye(3)
        self.x = None
        self.P = None

    def init(self, X):
        self.x = np.zeros(6)
        self.x[:3] = X
        self.P = np.eye(6)

    def predict(self):
        if self.x is None:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R=None):
        if self.x is None:
            self.init(z)
            return
        Rm = self.R if R is None else R
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + Rm
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

def filter_tracks(frames_sorted: List[int],
                  per_frame_points: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    # per_frame_points[frame][entity_key] = X
    # entity_key examples: "ball", "referee_0", "referee_1", "player_0"... up to counts
    kfs: Dict[str, ConstantVelocityKF] = {}
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    prev_frame = None
    for f in frames_sorted:
        out[f] = {}
        dt = DT_DEFAULT if prev_frame is None else max(DT_DEFAULT, (f - prev_frame) * DT_DEFAULT)
        prev_frame = f
        # predict all
        for key,kf in kfs.items():
            kf.dt = dt
            for i in range(3):
                kf.F[i, i+3] = dt
            kf.Q = 1.0 * np.diag([dt**4/4, dt**4/4, dt**4/4, dt**2, dt**2, dt**2])
            kf.predict()
        # update with observations
        for key,X in per_frame_points.get(f, {}).items():
            if key not in kfs:
                kfs[key] = ConstantVelocityKF(dt=dt)
                kfs[key].init(X)
            else:
                kfs[key].update(X)
            est = kfs[key].x.copy()
            out[f][key] = {
                "x": est[0],
                "y": est[1],
                "z": est[2],
                "vx": est[3],
                "vy": est[4],
                "vz": est[5],
                "observed": True
            }
        # for keys not observed, keep predictions
        for key,kf in kfs.items():
            if key not in out[f]:
                est = kf.x.copy()
                out[f][key] = {
                    "x": est[0],
                    "y": est[1],
                    "z": est[2],
                    "vx": est[3],
                    "vy": est[4],
                    "vz": est[5],
                    "observed": False
                }
    return out

# -------------------------------
# Main triangulation loop
# -------------------------------

def main():
    # Load
    tracks = load_all_tracks()
    calibs = load_all_calibs()

    # Build union of frames
    frames_sorted = frame_union(tracks)

    # Target counts per class
    target_counts = {
        "ball": 1,
        "referee": 2,
        "player": 12,
    }

    per_frame_points: Dict[int, Dict[str, np.ndarray]] = {}
    meta_errors: Dict[int, Dict[str, Any]] = {}

    for f in frames_sorted:
        # Build per-class observations
        per_class_obs = build_observation_sets(f, tracks, calibs)
        # Triangulate class by class
        out_entities: Dict[str, np.ndarray] = {}
        out_meta = {}

        # Ball: use all detections labeled "ball" if present, else allow "unassigned" candidates
        for label, tgt in [("ball", 1), ("referee", 2), ("player", 12)]:
            # Filter observations by matching label, plus for ball fallback to unassigned
            obs_list = []
            if label in per_class_obs:
                obs_list.extend(per_class_obs[label])
            if label == "ball" and "unassigned" in per_class_obs:
                obs_list.extend(per_class_obs["unassigned"])
            if len(obs_list) == 0:
                # no obs this frame
                continue
            # Build groups up to target_count
            # Construct a synthetic tracks dict just for this label for grouping
            # Simpler: use all obs_list directly and greedy-split: try k-means in image space per camera is complex; use sequential packing
            # Here we reuse group_entities_within_class with a faux tracks view:
            faux_tracks = {ck: {f: []} for ck in tracks.keys()}
            # reconstruct dets for this frame/label from obs_list by projecting back impossible; instead, group via a simple greedy approach:
            # We'll greedily form up to tgt groups by picking seeds and assigning obs that triangulate within threshold to that seed.
            remaining = obs_list.copy()
            groups = []
            while remaining and len(groups) < target_counts[label]:
                # pick a seed pair if possible
                if len(remaining) >= 2:
                    best_pair = None
                    best_err = 1e9
                    for i in range(len(remaining)):
                        for j in range(i+1, len(remaining)):
                            X = linear_triangulate_pair(remaining[i][0], remaining[i][1], remaining[i][2], remaining[i][3],
                                                        remaining[j][0], remaining[j][1], remaining[j][2], remaining[j][3])
                            if X is None:
                                continue
                            e = np.mean(reproj_errors(X, [remaining[i], remaining[j]]))
                            if e < best_err:
                                best_err = e
                                best_pair = (i,j,X)
                    if best_pair is None:
                        # take single as degenerate
                        seed_group = [remaining.pop(0)]
                        groups.append(seed_group)
                    else:
                        i,j,Xseed = best_pair
                        # collect compatible obs
                        seed_group = []
                        keep = []
                        for k,ob in enumerate(remaining):
                            e = np.mean(reproj_errors(Xseed, [ob]))
                            if e <= REPROJ_THRESH_PX * 2.0:
                                seed_group.append(ob)
                            else:
                                keep.append(ob)
                        remaining = keep
                        groups.append(seed_group)
                else:
                    groups.append([remaining.pop(0)])

            # Triangulate each group
            ent_idx = 0
            for g in groups:
                tri = triangulate_group(g)
                if tri is None:
                    continue
                X, used_idx, res_ref = tri
                key = f"{label}" if target_counts[label] == 1 else f"{label}_{ent_idx}"
                out_entities[key] = X
                out_meta[key] = {
                    "cams_used": len(used_idx),
                    "rmse": float(np.sqrt(np.mean(np.array(res_ref)**2))),
                }
                ent_idx += 1

        if out_entities:
            per_frame_points[f] = out_entities
            meta_errors[f] = out_meta

    # Temporal filtering
    filtered = filter_tracks(frames_sorted, per_frame_points)

    # Write CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["frame","entity","x","y","z","vx","vy","vz","observed","cams_used","rmse"])
        for f in frames_sorted:
            ent_map = filtered.get(f, {})
            meta_map = meta_errors.get(f, {})
            for key,vals in ent_map.items():
                me = meta_map.get(key, {})
                w.writerow([
                    f, key,
                    vals["x"], vals["y"], vals["z"],
                    vals["vx"], vals["vy"], vals["vz"],
                    int(vals["observed"]),
                    me.get("cams_used", 0),
                    me.get("rmse", float("nan"))
                ])

    print(f"Wrote {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
