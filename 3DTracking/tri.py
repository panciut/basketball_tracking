import json, csv, os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment, least_squares
from collections import Counter

TRACKING_RESULTS_RECT_DIR = "output/"
TRACKING_RESULTS_DIST_DIR = "../2DTracking/output/"
CALIBRATION_DIR = "../camera_data/"
OUTPUT_CSV = "output/3D_positions.csv"

LABELS_TO_TRIANGULATE = ["referee", "player", "unassigned", "ball"]
CAM_IDS = ["2", "4", "13"]

# Enhanced parameters
MAX_COST_MATCHING = 0.9
ADAPTIVE_REPROJ_BASE = 70.0
RANSAC_ERROR_THRESHOLD = 10.0
MIN_INLIER_RATIO = 0.6
CONSISTENCY_THRESHOLD = 0.7
SIGMA_PIXELS = 1.5

WEIGHT_2D = 0.5
WEIGHT_EPIPOLAR = 1.0

T_IN_MM = False

def _first_vec3(v) -> np.ndarray:
    arr = np.array(v, dtype=float)
    if arr.ndim == 1 and arr.size >= 3:
        return arr[:3].reshape(3, 1)
    if arr.ndim == 2:
        if arr.shape == (3, 1):
            return arr
        if arr.shape == (1, 3):
            return arr.reshape(3, 1)
        if arr.shape[0] >= 3 and arr.shape[1] >= 1:
            return arr[:3, :1]
        if arr.shape[1] >= 3 and arr.shape[0] >= 1:
            return arr[:1, :3].reshape(3, 1)
    if arr.ndim == 3:
        if arr.shape[0] >= 1 and arr.shape[1:] == (3, 1):
            return arr[0]
        if arr.shape[0] >= 1 and arr.shape[1:] == (1, 3):
            return arr[0].reshape(3, 1)

def _get_KRt_from_calib(calib: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = None
    for k in ["K", "camera_matrix", "mtx"]:
        if k in calib:
            K = np.array(calib[k], dtype=np.float32).reshape(3, 3)
            break
    if K is None:
        raise KeyError("Camera matrix not found in calibration data.")

    R = None
    if "R" in calib:
        R = np.array(calib["R"], dtype=np.float32).reshape(3, 3)
    elif "rvec" in calib:
        rvec = _first_vec3(calib["rvec"])
        R, _ = cv2.Rodrigues(rvec)
    elif "rvecs" in calib:
        rvec = _first_vec3(calib["rvecs"])
        R, _ = cv2.Rodrigues(rvec)
    else:
        raise KeyError("Rotation matrix or vector not found in calibration data.")

    t = None
    if "t" in calib:
        t = _first_vec3(calib["t"])
    elif "tvec" in calib:
        t = _first_vec3(calib["tvec"])
    elif "tvecs" in calib:
        t = _first_vec3(calib["tvecs"])
    else:
        raise KeyError("Translation vector not found in calibration data.")
    if T_IN_MM:
        t /= 1000.0
    return K, R, t

def _get_image_size(metadata_path: str) -> Tuple[int, int]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))["imsize"]

def load_calibration(calib_path: str) -> Dict[str, dict]:
    calib: Dict[str, dict] = {}
    root = Path(calib_path)
    camera_dirs = sorted(
        [d for d in root.iterdir() if (d / "calib" / "camera_calib.json").exists()]
    )
    if camera_dirs:
        for cam_dir in camera_dirs:
            cam_id = cam_dir.name
            calib_file = cam_dir / "calib" / "camera_calib.json"
            data = json.loads(
                (cam_dir / "calib" / "camera_calib.json").read_text(encoding="utf-8")
            )
            K, R, t = _get_KRt_from_calib(data)
            img_size = _get_image_size(cam_dir / "metadata.json")
            Rt = np.hstack([R, t])
            P = K @ Rt
            calib[cam_id] = {"K": K, "R": R, "t": t, "P": P, "img_size": img_size}
        return calib
    else:
        raise FileNotFoundError(f"No camera directories found in {calib_path}")

def _get_xy_from_bbox(bbox: List[float], label: str) -> Tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("Bounding box must have exactly four elements")
    x0, y0, a, b = bbox
    is_xyxy = False
    if (a - x0) > 0 and (b - y0) > 0:
        is_xyxy = True
    if a < 0 or b < 0:
        is_xyxy = True
    if is_xyxy:
        xc = (x0 + a) / 2.0
        yc = (y0 + b) / 2.0
    else:
        w, h = a, b
        if w <= 0 or h <= 0:
            xc = (x0 + a) / 2.0
            yc = (y0 + b) / 2.0
        else:
            xc = x0 + w / 2.0
            yc = y0 + h / 2.0
    return float(xc), float(yc)

def _parse_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(det, dict):
        raise ValueError("Detection entry is not a dictionary.")
    required_keys = {"id", "bbox", "label"}
    if not required_keys.issubset(det.keys()):
        raise ValueError(
            f"Detection entry missing required keys: {required_keys - det.keys()}"
        )
    x, y = _get_xy_from_bbox(det["bbox"], det["label"])
    return {"id": det["id"], "x": x, "y": y, "label": det["label"]}

def load_observations(path: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    root = Path(path)
    observations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    tracking_files = sorted(root.glob("tracking_results_rect_out*.json"))
    if not tracking_files:
        raise FileNotFoundError(f"No tracking result files found in {path}")
    for file in tracking_files:
        cam_id = file.stem.split("out")[-1]
        if not cam_id:
            raise ValueError(
                f"Camera ID could not be extracted from filename: {file.name}"
            )
        data = json.loads(file.read_text(encoding="utf-8"))
        for frame_str, detections in data.items():
            frame_id = str(frame_str)
            if frame_id not in observations:
                observations[frame_id] = {}
            if cam_id not in observations[frame_id]:
                observations[frame_id][f"cam_{cam_id}"] = []
            for det in detections:
                parsed_det = _parse_detection(det)
                if parsed_det["label"] in LABELS_TO_TRIANGULATE:
                    observations[frame_id][f"cam_{cam_id}"].append(parsed_det)
    return observations

def _get_fundamental_matrix(
    cam1: Dict[str, np.ndarray], cam2: Dict[str, np.ndarray]
) -> np.ndarray:
    R_rel = cam2["R"] @ cam1["R"].T
    t1 = cam1["t"].reshape(3)
    t2 = cam2["t"].reshape(3)
    t_rel = t2 - R_rel @ t1

    tx, ty, tz = float(t_rel[0]), float(t_rel[1]), float(t_rel[2])
    t_x = np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)

    E = t_x @ R_rel
    K1 = cam1["K"].astype(np.float64)
    K2 = cam2["K"].astype(np.float64)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def _sampson_distance(
    x1: float, y1: float, x2: float, y2: float, F: np.ndarray
) -> float:
    """Compute Sampson distance as a pure float (no 1x1 ndarray).

    Restituisce un float per evitare deprecation / problemi di broadcasting
    quando viene usato in cost_matrix e nelle liste di consistency.
    """
    p1 = np.array([x1, y1, 1.0], dtype=np.float64).reshape(3, 1)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64).reshape(3, 1)
    Fx1 = F @ p1           # (3,1)
    Ftx2 = F.T @ p2        # (3,1)
    denom = float(Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2)
    if denom < 1e-12:
        return 1e6
    num = float((p2.T @ F @ p1) ** 2)  # (1,1) -> float
    return num / denom

def adaptive_reproj_error_threshold(num_cams_used, base_threshold=ADAPTIVE_REPROJ_BASE):
    """Lower threshold for more cameras, higher for fewer cameras"""
    if num_cams_used >= 3:
        return base_threshold * 0.7  # Stricter for more cameras
    elif num_cams_used == 2:
        return base_threshold * 1.2  # More lenient for 2 cameras
    return base_threshold

def improved_pairwise_cost_matrix(
    dets_a: List[Dict[str, Any]],
    dets_b: List[Dict[str, Any]],
    F: np.ndarray,
    image_size: Tuple[int, int],
    calib_a: dict,
    calib_b: dict
) -> np.ndarray:
    W, H = image_size[0], image_size[1]
    norm = np.hypot(W, H)
    cost_matrix = np.full((len(dets_a), len(dets_b)), fill_value=0.0, dtype=np.float32)
    
    for i, a in enumerate(dets_a):
        for j, b in enumerate(dets_b):
            # 1. Label consistency cost
            label_cost = 0.0 if a["label"] == b["label"] else 1.0
            
            # 2. Epipolar distance
            d_epipolar = float(_sampson_distance(a["x"], a["y"], b["x"], b["y"], F))
            
            # 3. Normalized 2D distance
            d_2d = np.linalg.norm(np.array([a["x"] - b["x"], a["y"] - b["y"]])) / (norm + 1e-6)
            
            # 4. Geometric consistency (placeholder for motion model)
            geom_cost = 0.0
            
            # Combined cost with weights
            cost_matrix[i, j] = (
                WEIGHT_EPIPOLAR * d_epipolar + 
                WEIGHT_2D * d_2d +
                0.3 * label_cost +
                0.2 * geom_cost
            )
    
    # Enhanced normalization
    if cost_matrix.size:
        # Row normalization
        row_mins = np.min(cost_matrix, axis=1, keepdims=True)
        row_maxs = np.max(cost_matrix, axis=1, keepdims=True)
        row_range = row_maxs - row_mins
        row_range[row_range == 0] = 1.0
        cost_matrix_normalized_rows = (cost_matrix - row_mins) / row_range
        
        # Column normalization  
        col_mins = np.min(cost_matrix_normalized_rows, axis=0, keepdims=True)
        col_maxs = np.max(cost_matrix_normalized_rows, axis=0, keepdims=True)
        col_range = col_maxs - col_mins
        col_range[col_range == 0] = 1.0
        cost_matrix = (cost_matrix_normalized_rows - col_mins) / col_range
    
    return cost_matrix

def multi_view_consistency_check(group_nodes, observations, frame_id, calib, max_consistency_threshold=CONSISTENCY_THRESHOLD):
    """Check if a group is consistent across multiple views"""
    if len(group_nodes) < 3:
        return True  # Skip check for 2 views
        
    consistency_scores = []
    for i, (cam_i, det_i) in enumerate(group_nodes):
        for j, (cam_j, det_j) in enumerate(group_nodes):
            if i >= j:
                continue
                
            obs_i = observations[frame_id][cam_i][det_i]
            obs_j = observations[frame_id][cam_j][det_j]
            
            F = _get_fundamental_matrix(calib[cam_i], calib[cam_j])
            epipolar_dist = float(_sampson_distance(
                obs_i["x"], obs_i["y"], obs_j["x"], obs_j["y"], F
            ))
            
            # Normalize epipolar distance
            img_size = calib[cam_i]["img_size"]
            norm_factor = np.hypot(img_size[0], img_size[1])
            normalized_dist = float(epipolar_dist) / (norm_factor + 1e-6)
            score = 1.0 - min(normalized_dist, 1.0)
            consistency_scores.append(float(score))
    
    if not consistency_scores:
        return True
        
    mean_consistency = np.mean(consistency_scores)
    return mean_consistency >= max_consistency_threshold

def create_groups_from_matches(pairwise_matches):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ua, ub = find(a), find(b)
        if ua != ub:
            parent[ub] = ua

    for (ca, cb), pairs in pairwise_matches.items():
        for ma, mb in pairs:
            union(ma, mb)
    temp_groups = {}
    for node in list(parent.keys()):
        root = find(node)
        temp_groups.setdefault(root, []).append(node)
    groups = [
        {"group_id": f"C_{k:03d}", "nodes": nodes}
        for k, nodes in enumerate(temp_groups.values(), start=1)
    ]
    return groups

def _get_observation(
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    frame_id: str,
    cam_id: str,
    obj_id: int,
) -> Optional[Dict[str, Any]]:
    if frame_id not in data:
        return None
    if cam_id not in data[frame_id]:
        return None
    for obs in data[frame_id][cam_id]:
        if obs.get("id") == obj_id:
            return obs
    return None

def _linear_triangulate(
    Ps: List[np.ndarray], pos_list: List[Tuple[float, float]]
) -> np.ndarray:
    A = []
    for (u, v), P in zip(pos_list, Ps):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    X = X / (X[3] + 1e-12)
    return X 

def _project(K, R, t, X):
    Xc = R @ X.reshape(3,1) + t
    x = K @ Xc
    u = (x[0]/x[2]).item()
    v = (x[1]/x[2]).item()
    return np.array([u, v], dtype=float)

def ransac_triangulate(Ps, pos_list, num_iterations=50, error_threshold=RANSAC_ERROR_THRESHOLD):
    """RANSAC-based triangulation for outlier rejection"""
    if len(Ps) < 3:
        return _linear_triangulate(Ps, pos_list), list(range(len(Ps)))
    
    best_inliers = []
    best_X = None
    
    for _ in range(num_iterations):
        # Randomly select 2 views for minimal solution
        indices = np.random.choice(len(Ps), 2, replace=False)
        Ps_sample = [Ps[i] for i in indices]
        pos_sample = [pos_list[i] for i in indices]
        
        X_candidate = _linear_triangulate(Ps_sample, pos_sample)
        
        # Count inliers
        inliers = []
        for i, (P, (u, v)) in enumerate(zip(Ps, pos_list)):
            # Project and compute error
            x_proj = P @ X_candidate
            if x_proj[2] <= 0:  # Behind camera
                continue
                
            u_proj = x_proj[0] / x_proj[2]
            v_proj = x_proj[1] / x_proj[2]
            error = np.sqrt((u - u_proj)**2 + (v - v_proj)**2)
            
            if error < error_threshold:
                inliers.append(i)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_X = X_candidate
    
    # Final triangulation with all inliers
    if len(best_inliers) >= 2:
        Ps_inliers = [Ps[i] for i in best_inliers]
        pos_inliers = [pos_list[i] for i in best_inliers]
        return _linear_triangulate(Ps_inliers, pos_inliers), best_inliers
    
    return best_X, best_inliers if best_X is not None else (_linear_triangulate(Ps, pos_list), list(range(len(Ps))))

def _refine3dpoint_levenberg_marquardt(
    x0: np.ndarray,
    pos_list: List[Tuple[float, float]],
    cams_used: List[str],
    calib: Dict[str, dict],
) -> Tuple[np.ndarray, float, np.ndarray]:
    cams = [calib[cam_id] for cam_id in cams_used]
    cam_list = [
        (cams[i]["K"], cams[i]["R"], cams[i]["t"]) for i in range(len(pos_list))
    ]

    def residuals(x: np.ndarray) -> np.ndarray:
        res = []
        for (K,R,t), obs in zip(cam_list, pos_list):
            uhat = _project(K,R,t,x)
            res.extend(uhat - obs)
        return np.array(res)

    result = least_squares(residuals, x0, method="lm")
    x_opt = result.x
    cam_error = {}
    res = residuals(x_opt).reshape(-1, 2)
    for i, residual in enumerate(res):
        error = np.linalg.norm(residual)
        cam_error[f"cam_{i+1}"] = float(error)
    if result.jac is not None and result.jac.size > 0:
        jacobian = result.jac
    else:
        # Numerical jacobian fallback
        eps = 1e-6
        base = x_opt.copy()
        f0 = residuals(base)
        cols = []
        for k in range(3):
            p = base.copy()
            p[k] += eps
            f1 = residuals(p)
            cols.append((f1 - f0) / eps)
        jacobian = np.stack(cols, axis=1)
    return x_opt, cam_error, jacobian

def _compute_variance(jacobian: np.ndarray) -> np.ndarray:
    W = np.eye(jacobian.shape[0])
    jtj = jacobian.T @ W @ jacobian
    jtj += 1e-9 * np.eye(jtj.shape[0])
    cov = np.linalg.inv(jtj) * (SIGMA_PIXELS**2)
    return cov

def improved_triangulation_pipeline(calib, observations):
    points = {}
    
    for frame_id in observations:
        points[frame_id] = []
        cams_in_frame = sorted(observations[frame_id].keys())
        
        # Skip frames with too few cameras
        if len(cams_in_frame) < 2:
            continue
            
        pairwise_matches = {}
        for i in range(len(cams_in_frame)):
            for j in range(i + 1, len(cams_in_frame)):
                c1, c2 = cams_in_frame[i], cams_in_frame[j]
                Fa, Fb = calib[c1], calib[c2]
                F = _get_fundamental_matrix(Fa, Fb)
                
                C = improved_pairwise_cost_matrix(
                    observations[frame_id][c1],
                    observations[frame_id][c2],
                    F,
                    image_size=Fa["img_size"],
                    calib_a=Fa,
                    calib_b=Fb
                )
                
                if C.size == 0:
                    matches_id = []
                else:
                    r, c = linear_sum_assignment(C)
                    # Adaptive matching threshold based on number of detections
                    adaptive_threshold = MAX_COST_MATCHING * (1 + 0.1 * min(len(observations[frame_id][c1]), len(observations[frame_id][c2])))
                    matches_id = [
                        (int(q), int(p))
                        for p, q in zip(r, c)
                        if C[p, q] <= adaptive_threshold
                    ]
                
                matched = []
                for ma, mb in matches_id:
                    na = (c1, mb)
                    nb = (c2, ma)
                    matched.append((na, nb))
                pairwise_matches[(c1, c2)] = matched
        
        groups = create_groups_from_matches(pairwise_matches)
        
        for group in groups:
            Ps, pos_list, cams_used, dets_used = [], [], [], []
            for cam_id, det_id in group["nodes"]:
                obs = observations[frame_id][cam_id][det_id]
                if obs is not None:
                    Ps.append(calib[cam_id]["P"])
                    pos_list.append((obs["x"], obs["y"]))
                    cams_used.append(cam_id)
                    dets_used.append(observations[frame_id][cam_id][det_id])
            
            if len(cams_used) < 2:
                continue
                
            # Multi-view consistency check
            if not multi_view_consistency_check(group["nodes"], observations, frame_id, calib):
                continue
            
            # RANSAC triangulation
            X, inlier_indices = ransac_triangulate(Ps, pos_list)
            
            # Check inlier ratio
            inlier_ratio = len(inlier_indices) / len(group["nodes"])
            if inlier_ratio < MIN_INLIER_RATIO:
                continue
            
            # Use only inliers for refinement
            if len(inlier_indices) >= 2 and len(inlier_indices) < len(Ps):
                Ps = [Ps[i] for i in inlier_indices]
                pos_list = [pos_list[i] for i in inlier_indices]
                cams_used = [cams_used[i] for i in inlier_indices]
                dets_used = [dets_used[i] for i in inlier_indices]
            
            # Cheirality check
            cheir_ok = True
            for cam_id in cams_used:
                cam = calib[cam_id]
                R, t = cam["R"], cam["t"]
                Xc = R @ X[:3].reshape(3,1) + t
                if Xc[2,0] <= 0:
                    cheir_ok = False
                    break
            if not cheir_ok:
                continue
                
            x0 = X[:3]
            pos_fixed, cam_errors, jacobian = _refine3dpoint_levenberg_marquardt(
                x0, pos_list, cams_used, calib
            )
            cov = _compute_variance(jacobian)
            
            # Adaptive reprojection error threshold
            adaptive_threshold = adaptive_reproj_error_threshold(len(cams_used))
            mean_err = float(np.mean(list(cam_errors.values()))) if cam_errors else 1e9
            
            if mean_err > adaptive_threshold:
                continue
            
            # Label determination
            classes = [d.get("label") for d in dets_used if d.get("label")]
            if any(c == "ball" for c in classes):
                label = "ball"
            elif classes:
                label = Counter(classes).most_common(1)[0][0]
            else:
                label = "unknown"
                
            points[frame_id].append({
                "group_id": group["group_id"],
                "label": label,
                "X": pos_fixed.tolist(),
                "cov": cov.tolist(),
                "mean_reproj_error": mean_err,
                "cams_used": len(cams_used),
                "inliers_ratio": inlier_ratio
            })
    
    return points

def main():
    calib = load_calibration(CALIBRATION_DIR)
    observations = load_observations(TRACKING_RESULTS_RECT_DIR)
    points = improved_triangulation_pipeline(calib, observations)
    
    # Save to CSV
    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id","group_id","label","X","Y","Z","mean_reproj_error","cams_used","inliers_ratio"])
        for frame_id, pts in points.items():
            for p in pts:
                writer.writerow([
                    frame_id,
                    p["group_id"],
                    p["label"],
                    p["X"][0], p["X"][1], p["X"][2],
                    p["mean_reproj_error"],
                    p["cams_used"],
                    p["inliers_ratio"]
                ])
    
    total_points = sum(len(v) for v in points.values())
    print(f"Triangulated {total_points} 3D points. Saved to {OUTPUT_CSV}")
    
    # Print some statistics
    if total_points > 0:
        reproj_errors = [p["mean_reproj_error"] for frame_pts in points.values() for p in frame_pts]
        cams_used = [p["cams_used"] for frame_pts in points.values() for p in frame_pts]
        inlier_ratios = [p["inliers_ratio"] for frame_pts in points.values() for p in frame_pts]
        
        print(f"Average reprojection error: {np.mean(reproj_errors):.2f} pixels")
        print(f"Average cameras used: {np.mean(cams_used):.2f}")
        print(f"Average inlier ratio: {np.mean(inlier_ratios):.2f}")
        print(f"Frames with triangulated points: {len(points)}")
    
    return points

if __name__ == "__main__":
    main()