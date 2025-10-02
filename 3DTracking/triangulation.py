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

LABELS_TO_TRIANGULATE = ["referee", "player", "unassigned", "ball"]  # Labels to consider for triangulation (aggiunto 'ball')
CAM_IDS = ["2", "4", "13"]  # List of camera IDs to consider for triangulation

MAX_COST_MATCHING = 0.9  # Maximum cost for matching points across views
MAX_REPROJ_ERROR = 70.0  # Maximum allowable reprojection error in pixels
SIGMA_PIXELS = 1.5  # Assumed pixel noise standard deviation

WEIGHT_2D = 0.5
WEIGHT_EPIPOLAR = 1.0

T_IN_MM = False   # Whether the translation vectors are in millimeters

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
        # es. (N,3,1) o (N,1,3)
        if arr.shape[0] >= 1 and arr.shape[1:] == (3, 1):
            return arr[0]
        if arr.shape[0] >= 1 and arr.shape[1:] == (1, 3):
            return arr[0].reshape(3, 1)

def _get_KRt_from_calib(calib: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract the camera matrix K, rotation matrix R, and translation vector t from the calibration data
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
        t /= 1000.0  # Convert mm to meters
    return K, R, t


def _get_image_size(metadata_path: str) -> Tuple[int, int]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))["imsize"]


def load_calibration(calib_path: str) -> Dict[str, dict]:
    """
     Load camera calibration data from a JSON file.
     input: calib_path - path to folder of the calibration JSON file
     output: Dictionary with camera matrix and distortion coefficients and immage size
     {
        "cam_id" : {K,R,t,P,img_size}
    }
    """
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
    """Return center of bbox supporting xywh or xyxy formats.

    Formats:
      xywh: [x, y, w, h]
      xyxy: [x1, y1, x2, y2]

    Heuristic: if (a - x0) > 0 and (b - y0) > 0 treat as xyxy; if w/h <=0 fallback to xyxy.
    Supports negative coords (forces xyxy interpretation).
    """
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
    """
    Load 2D tracking observations from JSON files.
    input: path - path to folder of the tracking JSON files
    output: a dictionary with structure:
    {
        "0" : {
            "cam_2" : [ {id, x, y, label}, ... ] ,
            "cam_4" : [ {id, x, y, label}, ... ]
        },
        "1" : { ... }
    }
    where x,y are the 2D coordinates in the image plane
    """
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
    # cam dicts have K (3x3), R (3x3), t (3x1) in the same world frame.
    # Relative pose from cam1 to cam2:
    R_rel = cam2["R"] @ cam1["R"].T
    t1 = cam1["t"].reshape(3)  # ensure (3,)
    t2 = cam2["t"].reshape(3)  # ensure (3,)
    t_rel = t2 - R_rel @ t1  # also (3,)

    # Skew-symmetric of t_rel
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
    p1 = np.array([x1, y1, 1.0], dtype=np.float64).reshape(3, 1)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64).reshape(3, 1)
    Fx1 = F @ p1
    Ftx2 = F.T @ p2
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    if denom < 1e-12:
        return 1e6
    num = (p2.T @ F @ p1) ** 2
    return (num / denom).astype(float)


def pairwise_cost_matrix(
    dets_a: List[Dict[str, Any]],
    dets_b: List[Dict[str, Any]],
    F: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    W, H = image_size[0], image_size[1]
    norm = np.hypot(W, H)
    cost_matrix = np.full((len(dets_a), len(dets_b)), fill_value=0.0, dtype=np.float32)
    for i, a in enumerate(dets_a):
        for j, b in enumerate(dets_b):
            d_epipolar = _sampson_distance(a["x"], a["y"], b["x"], b["y"], F)
            d_2d = np.linalg.norm(np.array([a["x"] - b["x"], a["y"] - b["y"]])) / (
                norm + 1e-6
            )
            cost_matrix[i, j] = (WEIGHT_EPIPOLAR * d_epipolar + WEIGHT_2D * d_2d).item()

    if cost_matrix.size:
        l, h = np.percentile(cost_matrix, [5, 95])
        cost_matrix = (cost_matrix - l) / (max(h - l, 1e-6))
        cost_matrix = np.clip(cost_matrix, 0.0, 1.0)
    return cost_matrix


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
    """
    Retrieve a single observation by frame_id, cam_id, and object id.
    """
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
    # DLT over N views: stack 2 rows per view
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


def triangulation_pipeline(
    calib: Dict[str, dict], observations: Dict[str, Dict[str, List[Dict[str, Any]]]]
):
    points = {}
    for frame_id in observations:
        points[frame_id] = []
        cams_in_frame = sorted(observations[frame_id].keys())
        pairwise_matches = {}
        for i in range(len(cams_in_frame)):
            for j in range(i + 1, len(cams_in_frame)):
                c1, c2 = cams_in_frame[i], cams_in_frame[j]
                Fa, Fb = calib[c1], calib[c2]
                F = _get_fundamental_matrix(Fa, Fb)
                C = pairwise_cost_matrix(
                    observations[frame_id][c1],
                    observations[frame_id][c2],
                    F,
                    image_size=Fa["img_size"],
                )
                if C.size == 0:
                    matches_id = []
                else:
                    r, c = linear_sum_assignment(C)
                    matches_id = [
                        (int(q), int(p))
                        for p, q in zip(r, c)
                        if C[p, q] <= MAX_COST_MATCHING
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
            X = _linear_triangulate(Ps, pos_list)
            # Cheirality check: ensure positive depth in all used cameras
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

            
            mean_err = float(np.mean(list(cam_errors.values()))) if cam_errors else 1e9
            if mean_err > MAX_REPROJ_ERROR:
                continue

            classes = [d.get("label") for d in dets_used if d.get("label")]
            if any(c == "ball" for c in classes):
                label = "ball"
            elif classes:
                label = Counter(classes).most_common(1)[0][0]
            else:
                label = "unknown"
            points[frame_id].append(
                {
                    "group_id": group["group_id"],
                    "label": label,
                    "X": pos_fixed.tolist(),
                    "cov": cov.tolist(),
                    "mean_reproj_error": mean_err,
                    "cams_used": len(cams_used),
                }
            )
    return points


def main():
    calib = load_calibration(CALIBRATION_DIR)
    observations = load_observations(TRACKING_RESULTS_RECT_DIR)
    points = triangulation_pipeline(calib, observations)
    # Save to CSV
    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id","group_id","label","X","Y","Z","mean_reproj_error","cams_used"])
        for frame_id, pts in points.items():
            for p in pts:
                writer.writerow([
                    frame_id,
                    p["group_id"],
                    p["label"],
                    p["X"][0], p["X"][1], p["X"][2],
                    p["mean_reproj_error"],
                    p["cams_used"],
                ])
    print(f"Triangulated {sum(len(v) for v in points.values())} 3D points. Saved to {OUTPUT_CSV}")
    return points

if __name__ == "__main__":
    main()
