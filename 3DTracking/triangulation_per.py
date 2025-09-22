import os
import json
import csv
import math
from typing import Dict, List, Tuple
import numpy as np
import cv2

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TRACK_FILES = {
    "cam_2": os.path.join(OUTPUT_DIR, "tracking_results_out2.json"),
    "cam_4": os.path.join(OUTPUT_DIR, "tracking_results_out4.json"),
    "cam_13": os.path.join(OUTPUT_DIR, "tracking_results_out13.json"),
}
CALIB_DIRS = {
    "cam_2": os.path.join(BASE_DIR, ".." , "camera_data", "cam_2", "calib" , "camera_calib.json"),
    "cam_4": os.path.join(BASE_DIR, ".." , "camera_data", "cam_4", "calib" , "camera_calib.json"),
    "cam_13": os.path.join(BASE_DIR, ".." , "camera_data", "cam_13", "calib" , "camera_calib.json"),
}
RESULT_CSV = os.path.join(OUTPUT_DIR, "triangulated_3d.csv")

# ----------------------------
# Utilities
# ----------------------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5*(x1+x2), 0.5*(y1+y2))

def build_camera_matrices(calib_json: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns K, dist, R, t, P where:
    - K: 3x3 intrinsics
    - dist: distortion coefficients shape (1,5)
    - R: 3x3 rotation matrix (world->cam or cam->world? We assume rvec/tvec are world->cam extrinsics from OpenCV calibration)
    - t: 3x1 translation
    - P: 3x4 projection matrix  P = K [R|t]
    """
    K = np.array(calib_json["mtx"], dtype=np.float64)
    dist = np.array(calib_json["dist"], dtype=np.float64).reshape(1, -1)
    rvec = np.array(calib_json["rvecs"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(calib_json["tvecs"], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])
    return K, dist, R, tvec, P

def undistort_points_norm(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Undistort pixel points to normalized camera coordinates.
    pts: Nx2 pixel points
    returns: Nx2 normalized (x,y) in camera coordinates (z=1 along the optical axis).
    """
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist)  # returns normalized coordinates in ideal camera (Nx1x2)
    return undist.reshape(-1, 2)

def bearing_vectors_from_pixels(px: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given pixel centers, form 3D rays in world coordinates.
    Returns (C, dirs):
      - C: 3-vector camera center in world coords
      - dirs: Nx3 unit direction vectors in world coords passing through each pixel.
    """
    # Camera center in world: C = -R^T t
    C = -R.T @ t
    norm_xy = undistort_points_norm(px, K, dist)  # Nx2
    # Directions in camera coords: [x, y, 1]
    dirs_cam = np.hstack([norm_xy, np.ones((norm_xy.shape[0], 1))])  # Nx3
    # Normalize per-row
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    # Transform to world coords: dir_world = R^T * dir_cam
    dirs_world = (R.T @ dirs_cam.T).T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=1, keepdims=True)
    return C.reshape(3), dirs_world

def triangulate_rays(ray_origins: List[np.ndarray], ray_dirs: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Linear least-squares point closest to multiple skew lines.
    Each ray: X = O_i + s_i * d_i.
    Solve for X minimizing sum ||(I - d_i d_i^T)(X - O_i)||^2
    Returns (X, mean_orthogonal_error).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for O, d in zip(ray_origins, ray_dirs):
        d = d / np.linalg.norm(d)
        I = np.eye(3)
        P = I - np.outer(d, d)  # projector onto plane orthogonal to d
        A += P
        b += P @ O
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    # Compute mean orthogonal distance to each ray
    errs = []
    for O, d in zip(ray_origins, ray_dirs):
        v = X - O
        dist = np.linalg.norm(np.cross(v, d))  # |v x d| = distance times |d| (|d|=1)
        errs.append(dist)
    return X, float(np.mean(errs))

def reprojection_error_mm(Xw: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray, px_obs: np.ndarray) -> float:
    """
    Reproject 3D point to image and compute pixel error; if an approximate ground sample distance is known,
    convert to mm. Here we return pixel RMS across views; without homography to field, keep pixels.
    """
    Xw_h = np.hstack([Xw, 1.0])
    Xc = R @ Xw + t  # 3x1
    x = (K @ Xc).ravel()
    uv = x[:2] / x[2]
    # Distortion re-add for parity with observations (already distorted)
    uv_und = uv.reshape(1,1,2).astype(np.float64)
    uv_dist = cv2.projectPoints(Xw.reshape(1,1,3), cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1,2)
    return float(np.linalg.norm(uv_dist.ravel() - px_obs.ravel()))

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter + 1e-9
    return inter / union

def greedy_match_by_iou(objs_a, objs_b, min_iou=0.05):
    """
    objs_a/b are lists of dict with keys: id, bbox, label.
    Return list of pairs (a_idx, b_idx).
    """
    pairs = []
    used_b = set()
    for i, a in enumerate(objs_a):
        best = (-1, -1.0)
        for j, b in enumerate(objs_b):
            if b["label"] != a["label"] or j in used_b:
                continue
            s = iou(a["bbox"], b["bbox"])
            if s > best[1]:
                best = (j, s)
        if best[1] >= min_iou and best[0] >= 0:
            pairs.append((i, best[0]))
            used_b.add(best[0])
    return pairs

# ----------------------------
# Load data
# ----------------------------
def main():
    ensure_output_dir()

    # Load per-camera tracking
    tracks = {}
    for cam, path in TRACK_FILES.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing tracking file for {cam}: {path}")
        tracks[cam] = load_json(path)

    # Load per-camera calibration
    calibs = {}
    for cam, path in CALIB_DIRS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing calibration file for {cam}: {path}")
        calib_json = load_json(path)
        K, dist, R, t, P = build_camera_matrices(calib_json)
        calibs[cam] = {"K": K, "dist": dist, "R": R, "t": t, "P": P}

    # Gather all frame keys
    frame_keys = set()
    for cam in tracks:
        frame_keys |= set(tracks[cam].keys())
    # Sort numerically if possible
    try:
        frame_keys = sorted(frame_keys, key=lambda k: int(k))
    except:
        frame_keys = sorted(frame_keys)

    # Triangulation parameters
    classes_to_keep = {"player", "referee", "ball"}
    min_views = 2  # at least two cameras needed
    min_iou_crossview = 0.05
    max_ray_median_dist = 3000.0  # in mm; relax/tighten based on scene scale

    rows = []
    for fk in frame_keys:
        # Collect per-camera detections for this frame
        per_cam = {}
        for cam in ["cam_2", "cam_4", "cam_13"]:
            dets = tracks.get(cam, {}).get(fk, [])
            dets = [d for d in dets if d.get("label") in classes_to_keep]
            per_cam[cam] = dets

        # Pairwise greedy matches between cam_2-cam_4 and then try to extend to cam_13
        a = per_cam["cam_2"]
        b = per_cam["cam_4"]
        c = per_cam["cam_13"]

        pairs_ab = greedy_match_by_iou(a, b, min_iou=min_iou_crossview)

        # For each ab pair, find best c with same class by IoU with either a or b
        triplets = []
        used_c = set()
        for ia, ib in pairs_ab:
            la = a[ia]["label"]
            # candidates in c with same class
            cand_c = [(ic, iou(a[ia]["bbox"], c[ic]["bbox"]) + iou(b[ib]["bbox"], c[ic]["bbox"]))
                      for ic in range(len(c)) if c[ic]["label"] == la and ic not in used_c]
            ic_best = -1
            best_s = -1.0
            for ic, score in cand_c:
                if score > best_s:
                    best_s = score
                    ic_best = ic
            # Accept if some overlap exists; otherwise triangulate only from two views
            if ic_best >= 0 and best_s >= 2*min_iou_crossview:
                triplets.append((ia, ib, ic_best))
                used_c.add(ic_best)
            else:
                triplets.append((ia, ib, None))

        # Also handle detections that only exist in (cam_2, cam_13) or (cam_4, cam_13) if missed above
        # cam_2 vs cam_13
        remaining_a = [i for i in range(len(a)) if all(i != ia for ia,_,_ in triplets)]
        remaining_c = [i for i in range(len(c)) if i not in used_c]
        pairs_ac = greedy_match_by_iou([a[i] for i in remaining_a], [c[i] for i in remaining_c], min_iou=min_iou_crossview)
        for pa, pc in pairs_ac:
            ia = remaining_a[pa]
            ic = remaining_c[pc]
            triplets.append((ia, None, ic))
            used_c.add(ic)
        # cam_4 vs cam_13
        remaining_b = [i for i in range(len(b)) if all(i != ib for _,ib,_ in triplets if ib is not None)]
        remaining_c2 = [i for i in range(len(c)) if i not in used_c]
        pairs_bc = greedy_match_by_iou([b[i] for i in remaining_b], [c[i] for i in remaining_c2], min_iou=min_iou_crossview)
        for pb, pc in pairs_bc:
            ib = remaining_b[pb]
            ic = remaining_c2[pc]
            triplets.append((None, ib, ic))
            used_c.add(ic)

        # Triangulate each set
        for tpl in triplets:
            ia, ib, ic = tpl
            views = []
            meta = []
            for cam, idx in [("cam_2", ia), ("cam_4", ib), ("cam_13", ic)]:
                if idx is None:
                    continue
                det = per_cam[cam][idx]
                cx, cy = bbox_center(det["bbox"])
                views.append((cam, np.array([cx, cy], dtype=np.float64), det))
                meta.append((cam, det["id"], det["label"]))

            if len(views) < min_views:
                continue

            # All labels should match (skip if mismatch)
            labels = set(m[2] for m in meta)
            if len(labels) != 1:
                continue
            obj_label = list(labels)[0]

            # Build rays
            ray_origins = []
            ray_dirs = []
            for cam, pix, det in views:
                K = calibs[cam]["K"]; dist = calibs[cam]["dist"]
                R = calibs[cam]["R"]; t = calibs[cam]["t"]
                C, dirs = bearing_vectors_from_pixels(pix.reshape(1,2), K, dist, R, t)
                ray_origins.append(C)
                ray_dirs.append(dirs[0])

            # Outlier rejection: try all subsets of size >=2, pick with lowest mean ray distance
            best_X = None
            best_err = 1e18
            best_subset = None
            n = len(ray_dirs)
            # Enumerate subsets of views (simple: try all triples then pairs)
            subsets = []
            if n == 3:
                subsets.append([0,1,2])
            if n >= 2:
                subsets.extend([[0,1],[0,2],[1,2]][:max(0, 3 if n==3 else 1)])
                if n > 3:
                    # generic fallback: first 3, then all pairs
                    subsets.append(list(range(3)))
            for idxs in subsets:
                X, err = triangulate_rays([ray_origins[i] for i in idxs],
                                          [ray_dirs[i] for i in idxs])
                if err < best_err:
                    best_err = err
                    best_X = X
                    best_subset = idxs

            if best_X is None:
                continue

            if best_err > max_ray_median_dist:
                # reject noisy solution
                continue

            # Optional: compute average pixel reprojection residual for used views
            reproj_res = []
            for s in best_subset:
                cam, pix, _ = views[s]
                K = calibs[cam]["K"]; dist = calibs[cam]["dist"]
                R = calibs[cam]["R"]; t = calibs[cam]["t"]
                # Use cv2.projectPoints for accuracy with distortion
                img_pts, _ = cv2.projectPoints(best_X.reshape(1,3), cv2.Rodrigues(R)[0], t, K, dist)
                reproj_res.append(np.linalg.norm(img_pts.reshape(2) - pix))
            reproj_rms = float(np.sqrt(np.mean(np.square(reproj_res)))) if reproj_res else float('nan')

            # Collect IDs by camera for bookkeeping
            id2 = next((mid for cam,idv,lab in meta if cam=="cam_2" for mid in [idv] if True), -1)
            id4 = next((mid for cam,idv,lab in meta if cam=="cam_4" for mid in [idv] if True), -1)
            id13 = next((mid for cam,idv,lab in meta if cam=="cam_13" for mid in [idv] if True), -1)

            rows.append({
                "frame": fk,
                "class": obj_label,
                "track_id_cam2": id2 if any(cam=="cam_2" for cam,_,_ in views) else -1,
                "track_id_cam4": id4 if any(cam=="cam_4" for cam,_,_ in views) else -1,
                "track_id_cam13": id13 if any(cam=="cam_13" for cam,_,_ in views) else -1,
                "X": float(best_X[0]),
                "Y": float(best_X[1]),
                "Z": float(best_X[2]),
                "reproj_err_px": reproj_rms
            })

    # Save CSV
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame","class","track_id_cam2","track_id_cam4","track_id_cam13","X","Y","Z","reproj_err_px"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved {len(rows)} triangulated points to {RESULT_CSV}")

if __name__ == "__main__":
    main()