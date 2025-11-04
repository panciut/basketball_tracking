import json
import os
import re
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BASE_PATH = "./"
TRACKS3D_JSON = "output/tracking3d_output.json"
COCO_GT_PATH = "../data/rectified/_annotations_rectified.coco.json"
CAMERA_DATA = "../camera_data"
OUT_JSON = "output/metrics_summary.json"

# Timeline mapping: t_pred = FRAME_OFFSET + FRAME_SCALE * f_gt
FRAME_SCALE = 5  # GT 5 fps vs pred 25 fps ⇒ 5
FRAME_OFFSET = -2  # ultimo affine scelto dopo verifica

# Valutazione detection a soglie (metri)
PREC_THRESH_M = [0.3, 0.5, 1.0]

# Gate di matching
MATCH_GATE_M = 1.0
MATCH_GATE_BY_CLASS = {
    0: 1.8,  # ball (più ampio: può essere in aria, centro bbox)
    1: 1.2,  # player
    2: 1.2,  # referee
}

# Dedup GT proiettato (metri)
GT_MERGE_RADIUS_M = 0.8
MERGE_RADIUS_BY_CLASS = {0: 0.30, 1: 0.80, 2: 0.80}

# Intrinseche rettificate
ALPHA_RECT = 0.0
DEFAULT_IMG_SIZE = (3840, 2160)

# Unità di t nelle calibrazioni
ASSUME_T_MM = False

# FPS delle predizioni (per velocità e smoothing)
FPS_TRACKS = 25.0

# === Post-processing predizioni ===
ENABLE_STITCH = True
STITCH_TIME_GATE = 4  # max 4 frame di gap tra fine A e inizio B
STITCH_DIST_GATE_BY_CLS = {0: 1.2, 1: 0.8, 2: 0.8}  # distanza al join (m)
STITCH_REQUIRE_DIR = True  # richiede coerenza direzione
STITCH_DIR_MAX_DEG = 45.0  # max angolo tra direzioni A e B
STITCH_SPEED_RATIO_MAX = 2.0  # rapporti di modulo velocità
STITCH_OVERLAP_FORBID = True  # nessun overlap temporale
STITCH_CONFLICT_RADIUS = 1.0  # nessun altro pred vicino al join entro 1 m

MIN_TRACK_LEN_FRAMES = 5  # filtro lunghezza minima

ENABLE_SPEED_FILTER = True
SPEED_MAX_BY_CLASS = {  # velocità plausibili (m/s)
    0: 35.0,  # ball (lanci)
    1: 9.0,  # player
    2: 8.0,  # referee
}
SPEED_TWO_STRIKE = True  # richiede 2 step consecutivi > soglia per rimozione

ENABLE_SMOOTH_ZL = True
ZL_WINDOW = 5  # finestra dispari del moving average centrato

# NMS metrico per frame/classe
ENABLE_PRED_NMS = True
PRED_NMS_RADIUS_BY_CLS = {0: 0.35, 1: 0.25, 2: 0.30}  # player meno soppressivo

# === Tracking GT (per metriche IDF1/MOTA) ===
ENABLE_TRACKING_METRICS = True
LINK_GATE_BY_CLASS = {0: 1.2, 1: 1.2, 2: 1.2}
LINK_MAX_FRAME_GAP = 1


# =========================
# CLASSI / MAPPING
# =========================
def gt_class_conversion(class_id: int) -> int:
    """COCO → {0:ball, 1:player, 2:referee}"""
    if class_id == 1:
        return 0
    elif class_id in (7, 8):
        return 2
    else:
        return 1


CLASS_ID2NAME = {0: "ball", 1: "player", 2: "referee"}
_CAM_RGX = re.compile(r"cam[_-]?(\d+)", re.IGNORECASE)


def pred_class_to_id(name: str) -> int:
    """Robusto: gestisce 0/1/2, stringhe IT/EN."""
    if name is None:
        return 1
    s = str(name).strip().lower()
    if s in {"0", "1", "2"}:
        return int(s)
    if any(k in s for k in ["ball", "palla"]):
        return 0
    if any(k in s for k in ["ref", "referee", "arbitro", "arb"]):
        return 2
    if any(k in s for k in ["player", "giocatore", "person", "human"]):
        return 1
    return 1


def normalize_cam_id(cam_raw: str) -> str:
    """out2/out_13/camera-4/cam4 → cam_<num>"""
    s = str(cam_raw).lower().replace("-", "_").strip()
    m = re.search(r"(?:cam|out|camera|c)?_?(\d{1,2})$", s) or re.search(
        r"(\d{1,2})$", s
    )
    return f"cam_{int(m.group(1))}" if m else s


# =========================
# CALIBRAZIONE / OMOLOGRAFIA
# =========================
def _as3x3(a) -> np.ndarray:
    return np.array(a, float).reshape(3, 3)


def parse_KRt(calib_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # K
    if "K" in calib_data:
        K = _as3x3(calib_data["K"])
    elif "mtx" in calib_data:
        K = _as3x3(calib_data["mtx"])
    else:
        fx = calib_data.get("fx", 1000.0)
        fy = calib_data.get("fy", 1000.0)
        cx = calib_data.get("cx", 320.0)
        cy = calib_data.get("cy", 240.0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    # R
    if "R" in calib_data:
        R = _as3x3(calib_data["R"])
    else:
        rvec = np.array(calib_data.get("rvecs", calib_data.get("rvec")), float).reshape(
            3, 1
        )
        R, _ = cv2.Rodrigues(rvec)
    # t
    if "t" in calib_data:
        t = np.array(calib_data["t"], float).reshape(3, 1)
    else:
        t = np.array(calib_data.get("tvecs", calib_data.get("tvec")), float).reshape(
            3, 1
        )
    units = (calib_data.get("units") or calib_data.get("t_units") or "").strip().lower()
    try:
        if units in ("mm", "millimeter", "millimeters"):
            t = t / 1000.0
        elif units in ("m", "meter", "meters"):
            pass
        else:
            if ASSUME_T_MM or float(np.linalg.norm(t)) > 2000.0:
                t = t / 1000.0
    except Exception:
        pass
    return K, R, t


def _collect_cam_image_sizes_from_coco(
    coco_json_path: str,
) -> Dict[str, Tuple[int, int]]:
    data = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    cam_sizes: Dict[str, Tuple[int, int]] = {}
    seen_cams: set[str] = set()
    for img in data.get("images", []):
        name = img.get("file_name") or (img.get("extra", {}) or {}).get("name") or ""
        _, _, cam = deduce_cam_and_frame_from_name(str(name))
        if not cam:
            continue
        cam = normalize_cam_id(cam)
        seen_cams.add(cam)
        w = img.get("width", None)
        h = img.get("height", None)
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            cam_sizes.setdefault(cam, (w, h))
    for cam in seen_cams:
        cam_sizes.setdefault(cam, DEFAULT_IMG_SIZE)
    return cam_sizes


def load_calibrations_and_H(
    camera_data_path: str,
    cam_image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    alpha: float = ALPHA_RECT,
) -> Dict[str, dict]:
    cams = {}
    for cam_dir in sorted(Path(camera_data_path).glob("*")):
        cam_key = normalize_cam_id(cam_dir.name)
        calib_file = cam_dir / "calib" / "camera_calib.json"
        if not calib_file.exists():
            continue
        j = json.loads(calib_file.read_text(encoding="utf-8"))
        K, R, t = parse_KRt(j)
        dist = None
        for k in ("dist", "D", "distCoeffs"):
            if k in j:
                dist = np.array(j[k], float).reshape(-1)
                break
        K_use = None
        for k in ("K_rect", "Knew", "K_undistorted", "intrinsic_rectified"):
            if k in j:
                try:
                    K_use = _as3x3(j[k])
                    break
                except Exception:
                    pass
        if K_use is None and dist is not None and cam_image_sizes is not None:
            wh = cam_image_sizes.get(cam_key, DEFAULT_IMG_SIZE)
            try:
                K_new, _ = cv2.getOptimalNewCameraMatrix(
                    K.astype(np.float32), dist.astype(np.float32), wh, alpha, wh
                )
                K_use = K_new.astype(float)
            except Exception:
                K_use = K
        if K_use is None:
            K_use = K
        H = K_use @ np.column_stack([R[:, 0], R[:, 1], t.reshape(3)])
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.eye(3, dtype=float)
        cams[cam_key] = {"K": K_use, "R": R, "t": t, "H": H, "H_inv": H_inv}
    if not cams:
        raise FileNotFoundError("Nessuna calibrazione trovata in camera_data.")
    return cams


def inv_homography_to_field_xy(
    H_or_cal: np.ndarray | dict, u: float, v: float
) -> Tuple[float, float]:
    Hi = (
        H_or_cal["H_inv"]
        if isinstance(H_or_cal, dict) and "H_inv" in H_or_cal
        else np.linalg.inv(H_or_cal)
    )
    q = Hi @ np.array([u, v, 1.0], float)
    return float(q[0] / q[2]), float(q[1] / q[2])


# =========================
# GT LOADER (COCO rettificato)
# =========================
def deduce_cam_and_frame_from_name(
    fname: str,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    base = Path(fname).name
    stem, _ = os.path.splitext(base)
    patterns = [
        r"(?i)(?P<video>.*?)[_/\\-]?(?P<cam>(?:cam|out|camera|c)_?\d{1,2}).*?(?:frame[_-]?|f[_-]?|_)?(?P<frame>\d+)",
        r"(?i)(?P<cam>(?:cam|out|camera|c)_?\d{1,2}).*?(?P<frame>\d+)",
        r"(?i)(?P<frame>\d+).*?(?P<cam>(?:cam|out|camera|c)_?\d{1,2})",
    ]
    video = cam = None
    frame = None
    for p in patterns:
        m = re.search(p, base)
        if m:
            video = (m.groupdict().get("video") or "").strip() or None
            cam = normalize_cam_id(m.group("cam"))
            frame = int(m.group("frame"))
            break
    if cam is None:
        parts = [p for p in str(fname).replace("\\", "/").split("/") if p]
        for token in reversed(parts[:-1]):
            n = normalize_cam_id(token)
            if n.startswith("cam_"):
                cam = n
                break
    if frame is None:
        m = re.search(r"(\d+)", stem)
        if m:
            frame = int(m.group(1))
    return video, frame, cam


def bbox_center_uv(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    return x + 0.5 * w, y + 0.5 * h


def bbox_bottom_center_uv(
    x: float, y: float, w: float, h: float
) -> Tuple[float, float]:
    return x + 0.5 * w, y + h


def load_gt_coco(
    coco_json_path: str
) -> Dict[Tuple[int, str], List[dict]]:
    """
    Ritorna: {(frame_idx, cam_id_norm): [ {'u','v','cls_id','bbox','original_class'} , ... ]}
    BALL (0): centro bbox; PLAYER/REFEREE (1/2): bottom-center.
    """
    data = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}
    gt = defaultdict(list)
    for a in data.get("annotations", []):
        img = images.get(a.get("image_id"))
        if not img:
            continue
        name = img.get("file_name") or (img.get("extra", {}) or {}).get("name") or ""
        video, fidx, cam_id = deduce_cam_and_frame_from_name(str(name))
        if cam_id is None or fidx is None:
            continue
        cam_id = normalize_cam_id(cam_id)
        cid = int(a.get("category_id", -1))
        cls_id = gt_class_conversion(cid)
        x, y, w, h = map(float, a.get("bbox", [0, 0, 0, 0]))
        if w <= 0 or h <= 0:
            continue
        if cls_id == 0:
            u, v = bbox_center_uv(x, y, w, h)
        else:
            u, v = bbox_bottom_center_uv(x, y, w, h)
        gt[(fidx, cam_id)].append(
            {
                "u": u,
                "v": v,
                "class_id": cls_id,
                "original_class": cid,
                "bbox": [x, y, x + w, y + h],
            }
        )
    return dict(gt)


# =========================
# ALLINEAMENTO TEMPORALE
# =========================
def align_gt_frames(
    gt_by_fc: Dict[Tuple[int, str], List[dict]], tracked_frames_available: set[int]
) -> Dict[Tuple[int, str], List[dict]]:
    aligned = {}
    for (fgt, cam), items in gt_by_fc.items():
        t = FRAME_OFFSET + FRAME_SCALE * int(fgt)
        if t in tracked_frames_available:
            aligned[(t, cam)] = items
    return aligned


# =========================
# PROIEZIONE GT SU CAMPO + DEDUP
# =========================
def project_gt_to_field(
    gt_by_t_cam: Dict[Tuple[int, str], List[dict]], calibrations: Dict[str, dict]
) -> Dict[int, List[dict]]:
    out = defaultdict(list)
    for (t, cam_id), anns in gt_by_t_cam.items():
        cal = calibrations.get(cam_id)
        if not cal:
            continue
        for a in anns:
            X, Y = inv_homography_to_field_xy(cal, a["u"], a["v"])
            rec = dict(a)
            rec["field_x"] = float(X)
            rec["field_y"] = float(Y)
            rec["cam_id"] = cam_id
            out[t].append(rec)
    return dict(out)


def _greedy_merge_points(points, radius):
    if not points:
        return []
    unused = list(range(len(points)))
    clusters = []
    while unused:
        i = unused.pop(0)
        cluster_idx = [i]
        changed = True
        while changed:
            changed = False
            cx = float(np.mean([points[k][0] for k in cluster_idx]))
            cy = float(np.mean([points[k][1] for k in cluster_idx]))
            to_add = [
                j
                for j in list(unused)
                if math.hypot(points[j][0] - cx, points[j][1] - cy) <= radius
            ]
            if to_add:
                for j in to_add:
                    unused.remove(j)
                    cluster_idx.append(j)
                changed = True
        cx = float(np.mean([points[k][0] for k in cluster_idx]))
        cy = float(np.mean([points[k][1] for k in cluster_idx]))
        clusters.append((cx, cy, cluster_idx))
    return clusters


def dedup_gt_per_frame(projected_gt_by_t, merge_radius_m=GT_MERGE_RADIUS_M):
    out = {}
    for t, anns in projected_gt_by_t.items():
        by_cls = defaultdict(list)
        for a in anns:
            by_cls[a["class_id"]].append(a)
        merged_all = []
        for cls_id, lst in by_cls.items():
            pts = [(a["field_x"], a["field_y"]) for a in lst]
            r = MERGE_RADIUS_BY_CLASS.get(cls_id, merge_radius_m)
            clusters = _greedy_merge_points(pts, r)
            for cx, cy, idxs in clusters:
                merged_all.append(
                    {
                        "field_x": cx,
                        "field_y": cy,
                        "class_id": cls_id,
                        "n_views": len(idxs),
                    }
                )
        out[t] = merged_all
    return out


# =========================
# PRED LOADER + POST-PROCESS
# =========================


def load_predictions(csv_or_json_path: str) -> pd.DataFrame:
    """Load predictions either from a CSV (t,track_id,class,x,y,z) or from a JSON
    produced by your 3D tracker (list of tracks with per-frame history).
    The resulting DataFrame has columns: t (int frame), track_id (int), class (str),
    cls_id (int), x, y, z (float). If a 'conf' score exists it is kept as 'score'.
    """
    path = str(csv_or_json_path)
    ext = Path(path).suffix.lower()
    if ext == ".json":
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        # expected format:
        # [ {"track_id": int, "label": "player|referee|ball", "confidence": float,
        #     "history": [ {"frame": int, "t": float, "x": [x,y,z], "interp": bool, "conf": float}, ...] }, ... ]
        rows = []
        for tr in j:
            tid = int(tr.get("track_id"))
            label = tr.get("label", "player")
            hist = tr.get("history") or []
            for h in hist:
                frame = (
                    int(h.get("frame"))
                    if "frame" in h
                    else int(round(float(h.get("t", 0.0)) * FPS_TRACKS))
                )
                xyz = h.get("x") or [None, None, None]
                if xyz is None or len(xyz) < 2:
                    continue
                x = float(xyz[0])
                y = float(xyz[1])
                z = float(xyz[2] if len(xyz) > 2 else 0.0)
                score = h.get("conf", None)
                rows.append(
                    {
                        "t": frame,
                        "track_id": tid,
                        "class": label,
                        "x": x,
                        "y": y,
                        "z": z,
                        "score": score,
                    }
                )
        if not rows:
            raise ValueError(
                "JSON appears empty or not in the expected format (no history rows)."
            )
        df = pd.DataFrame(rows)
        # normalize/massage types and add cls_id
        df["cls_id"] = df["class"].map(pred_class_to_id)
        df["t"] = df["t"].astype(int)
        for c in ("x", "y", "z"):
            df[c] = df[c].astype(float)
        df["track_id"] = df["track_id"].astype(int)
        return df.sort_values(["t", "cls_id", "track_id"]).reset_index(drop=True)
    else:
        # fallback to the original CSV loader
        df = pd.read_csv(path)
        required = {"t", "track_id", "class", "x", "y", "z"}
        if not required.issubset(df.columns):
            raise ValueError(
                "tracks3d.csv manca colonne richieste: t, track_id, class, x, y, z"
            )
        df["cls_id"] = df["class"].map(pred_class_to_id)
        df["t"] = df["t"].astype(int)
        for c in ("x", "y", "z"):
            df[c] = df[c].astype(float)
        df["track_id"] = df["track_id"].astype(int)
        return df


def _track_spans(df: pd.DataFrame) -> Dict[int, Tuple[int, int, int]]:
    spans = df.groupby("track_id")["t"].agg(["min", "max"])
    return {
        tid: (int(r["min"]), int(r["max"]), int(r["max"] - r["min"] + 1))
        for tid, r in spans.iterrows()
    }


def _velocities_per_track(df: pd.DataFrame) -> Dict[int, pd.Series]:
    vel = {}
    for tid, g in df.sort_values("t").groupby("track_id"):
        arr = g[["t", "x", "y"]].to_numpy()
        if len(arr) < 2:
            vel[tid] = pd.Series([], dtype=float)
            continue
        dt = np.diff(arr[:, 0]) / FPS_TRACKS
        dxy = np.sqrt(np.diff(arr[:, 1]) ** 2 + np.diff(arr[:, 2]) ** 2)
        s = np.divide(dxy, np.maximum(dt, 1e-6))
        vel[tid] = pd.Series(s, index=g["t"].iloc[1:].to_list(), dtype=float)
    return vel


def _smooth_zero_lag_centered(vals: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(vals) < 3:
        return vals.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    pad_left = vals[1 : pad + 1][::-1] if len(vals) > 1 else vals[:1]
    pad_right = vals[-pad - 1 : -1][::-1] if len(vals) > 1 else vals[-1:]
    ext = np.concatenate([pad_left, vals, pad_right])
    kernel = np.ones(win, dtype=float) / win
    sm = np.convolve(ext, kernel, mode="valid")  # zero-lag centrato
    return sm


def smooth_tracks_zero_lag(df: pd.DataFrame, win: int = ZL_WINDOW) -> pd.DataFrame:
    if not ENABLE_SMOOTH_ZL:
        return df
    out = []
    for tid, g in df.sort_values("t").groupby("track_id"):
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        xs = _smooth_zero_lag_centered(x, win)
        ys = _smooth_zero_lag_centered(y, win)
        gg = g.copy()
        gg["x"] = xs
        gg["y"] = ys
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def speed_filter_two_strike(df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_SPEED_FILTER:
        return df
    vel = _velocities_per_track(df)
    bad = set()
    for tid, v in vel.items():
        if v.empty:
            continue
        g = df[df["track_id"] == tid].sort_values("t")
        cls = int(g["cls_id"].mode().iat[0]) if not g.empty else 1
        vmax = SPEED_MAX_BY_CLASS.get(cls, 10.0)
        # two-strike: rimuovi frame consecutivi se due o più consecutivi > soglia
        over = v > vmax
        # trova run di consecutivi True
        run_len = 0
        for t_val, is_over in over.items():
            run_len = run_len + 1 if is_over else 0
            if is_over and run_len >= 2:
                bad.add((tid, t_val))  # segna il secondo (e successivi) in poi
    if not bad:
        return df
    mask = ~df.apply(lambda r: (r["track_id"], r["t"]) in bad, axis=1)
    kept = int(mask.sum())
    removed = int((~mask).sum())
    print(
        f"After speed filter (two-strike): kept {kept}/{len(df)} predictions, removed {removed}"
    )
    return df[mask].copy()


def metric_nms_framewise(
    df: pd.DataFrame, radii: Dict[int, float] = PRED_NMS_RADIUS_BY_CLS
) -> pd.DataFrame:
    if not ENABLE_PRED_NMS:
        return df
    spans = _track_spans(df)
    keep_rows = []
    for t, df_t in df.groupby("t"):
        for cls_id, group in df_t.groupby("cls_id"):
            r = radii.get(int(cls_id), 0.3)
            pts = group[["x", "y"]].to_numpy()
            idx = list(group.index)
            used = [False] * len(idx)
            while True:
                # scegli la "migliore" come quella con traccia più lunga
                best = None
                best_len = -1
                best_i = -1
                for k, ridx in enumerate(idx):
                    if used[k]:
                        continue
                    tid = int(group.loc[ridx, "track_id"])
                    length = spans.get(tid, (0, 0, 1))[2]
                    if length > best_len:
                        best_len = length
                        best = ridx
                        best_i = k
                if best is None:
                    break
                keep_rows.append(best)
                used[best_i] = True
                bx, by = float(group.loc[best, "x"]), float(group.loc[best, "y"])
                # sopprimi vicini entro r
                for k, ridx in enumerate(idx):
                    if used[k]:
                        continue
                    px, py = float(group.loc[ridx, "x"]), float(group.loc[ridx, "y"])
                    if math.hypot(px - bx, py - by) <= r:
                        used[k] = True
                # ciclo continua finché ci sono punti non usati
    kept_df = df.loc[keep_rows].copy().sort_values(["t", "cls_id", "track_id"])
    print("After metric NMS per frame/class:")
    _log_tracks_basic(kept_df)
    return kept_df


def _dir_vec(track_xy: np.ndarray, tail: int = 3) -> np.ndarray:
    """vettore direzione medio su ultimi/prime 3 step; ritorna unit vector o zero."""
    if track_xy.shape[0] < 2:
        return np.array([0.0, 0.0])
    tail = max(1, min(tail, track_xy.shape[0] - 1))
    v = track_xy[-tail:] - track_xy[-tail - 1 : -1]
    vec = v.mean(axis=0)
    n = np.linalg.norm(vec)
    return vec / max(n, 1e-9)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _no_conflict_at_join(
    df_t: pd.DataFrame, x: float, y: float, radius: float, ignore_tid: int
) -> bool:
    if df_t.empty:
        return True
    for r in df_t.itertuples(index=False):
        if int(r.track_id) == ignore_tid:
            continue
        if math.hypot(float(r.x) - x, float(r.y) - y) < radius:
            return False
    return True


def stitch_tracks(df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_STITCH:
        return df
    df = df.sort_values(["cls_id", "track_id", "t"]).copy()
    spans = _track_spans(df)
    next_id = max(df["track_id"]) + 1 if not df.empty else 0
    remap = {}  # track_id -> new_track_id

    for cls_id, gcls in df.groupby("cls_id"):
        # build candidate starts/ends
        ends = {}
        starts = {}
        for tid, g in gcls.groupby("track_id"):
            t0, t1 = int(g["t"].min()), int(g["t"].max())
            ends[tid] = g[g["t"] == t1][["x", "y", "t"]].iloc[-1].to_numpy()
            starts[tid] = g[g["t"] == t0][["x", "y", "t"]].iloc[0].to_numpy()

        # try to connect end(A) -> start(B)
        used_B = set()
        for tidA in sorted(ends.keys()):
            if tidA in remap:
                continue  # già rimappato
            xA, yA, tA = ends[tidA]
            # candidati B con inizio dopo A e gap ≤ STITCH_TIME_GATE
            cand = []
            for tidB in sorted(starts.keys()):
                if tidB == tidA or tidB in remap or tidB in used_B:
                    continue
                xB, yB, tB = starts[tidB]
                gap = int(tB - tA)
                if gap <= 0 or gap > STITCH_TIME_GATE:
                    continue
                # distanza al join
                dist_gate = STITCH_DIST_GATE_BY_CLS.get(int(cls_id), 0.8)
                if math.hypot(xB - xA, yB - yA) > dist_gate:
                    continue
                # vieta overlap temporale
                if STITCH_OVERLAP_FORBID:
                    t0A, t1A, _ = spans.get(int(tidA), (0, 0, 0))
                    t0B, t1B, _ = spans.get(int(tidB), (0, 0, 0))
                    if not (t1A < t0B):  # overlap
                        continue
                # coerenza direzione/velocità
                if STITCH_REQUIRE_DIR:
                    gA = gcls[gcls["track_id"] == tidA].sort_values("t")
                    gB = gcls[gcls["track_id"] == tidB].sort_values("t")
                    vA = _dir_vec(gA[["x", "y"]].to_numpy(), tail=3)
                    vB = _dir_vec(gB[["x", "y"]].to_numpy()[:4], tail=3)  # primi step
                    if np.linalg.norm(vA) < 1e-6 or np.linalg.norm(vB) < 1e-6:
                        continue
                    ang = _angle_deg(vA, vB)
                    if ang > STITCH_DIR_MAX_DEG:
                        continue
                    # velocità comparabili
                    vel = _velocities_per_track(pd.concat([gA, gB], ignore_index=True))
                    vA_mod = (
                        float(np.median(vel.get(tidA, pd.Series([0]))))
                        if not vel.get(tidA, pd.Series()).empty
                        else 0.0
                    )
                    vB_mod = (
                        float(np.median(vel.get(tidB, pd.Series([0]))))
                        if not vel.get(tidB, pd.Series()).empty
                        else 0.0
                    )
                    ratio = (
                        max(vA_mod, 1e-3) / max(vB_mod, 1e-3) if vB_mod > 0 else 999.0
                    )
                    if (
                        ratio > STITCH_SPEED_RATIO_MAX
                        or (1.0 / ratio) > STITCH_SPEED_RATIO_MAX
                    ):
                        continue
                # nessun conflitto con altri pred al frame di start B
                df_tB = gcls[gcls["t"] == int(tB)]
                if not _no_conflict_at_join(
                    df_tB, xB, yB, STITCH_CONFLICT_RADIUS, ignore_tid=int(tidB)
                ):
                    continue
                cand.append((gap, math.hypot(xB - xA, yB - yA), int(tidB)))
            if not cand:
                continue
            # scegli il B con gap minimo poi distanza minima
            cand.sort()
            tidB = cand[0][2]
            # unisci: remappa B in A (mantieni id A)
            remap[int(tidB)] = int(tidA)
            used_B.add(int(tidB))

    if not remap:
        return df

    df2 = df.copy()
    df2["track_id"] = df2["track_id"].apply(lambda tid: remap.get(int(tid), int(tid)))
    df2 = df2.sort_values(["cls_id", "track_id", "t"]).reset_index(drop=True)
    print("After stitching:")
    _log_tracks_basic(df2)
    return df2


def filter_min_length(
    df: pd.DataFrame, min_len: int = MIN_TRACK_LEN_FRAMES
) -> pd.DataFrame:
    if min_len <= 1:
        return df
    spans = _track_spans(df)
    keep_ids = {tid for tid, (_, _, L) in spans.items() if L >= min_len}
    kept = df[df["track_id"].isin(keep_ids)].copy()
    print(
        f"After min-length filter (≥{min_len} frames): kept {kept.shape[0]}/{df.shape[0]}"
    )
    _log_tracks_basic(kept)
    return kept


def _log_tracks_basic(df: pd.DataFrame):
    if df.empty:
        print("Pred tracks: count=0")
        return
    lens = df.groupby("track_id")["t"].agg(lambda s: int(s.max() - s.min() + 1))
    matched_len = lens  # placeholder se calcoli “matched frames per pred track”
    print(
        f"Pred tracks: count={len(lens)}, median_len={float(lens.median()):.1f} frames, P90={float(lens.quantile(0.9))}"
    )
    # matched frames per track: richiede GT, lo stampiamo in main dopo i match


# =========================
# MATCHING (Hungarian)
# =========================
def hungarian_match_xy(
    gt_xy: List[Tuple[float, float]],
    pr_xy: List[Tuple[float, float]],
    gate_m: float = MATCH_GATE_M,
):
    if not gt_xy or not pr_xy:
        return [], list(range(len(gt_xy))), list(range(len(pr_xy)))
    C = np.full((len(gt_xy), len(pr_xy)), 1e6, float)
    for i, (Xg, Yg) in enumerate(gt_xy):
        for j, (Xp, Yp) in enumerate(pr_xy):
            d = math.hypot(Xp - Xg, Yp - Yg)
            if d <= gate_m:
                C[i, j] = d
    r, c = linear_sum_assignment(C)
    matches = [(int(i), int(j), float(C[i, j])) for i, j in zip(r, c) if C[i, j] < 1e6]
    matched_gt = {i for i, _, _ in matches}
    matched_pr = {j for _, j, _ in matches}
    unmatched_gt = [i for i in range(len(gt_xy)) if i not in matched_gt]
    unmatched_pr = [j for j in range(len(pr_xy)) if j not in matched_pr]
    return matches, unmatched_gt, unmatched_pr


# =========================
# METRICHE (detection/posizione)
# =========================
def calc_detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calc_position_metrics(dists: List[float]) -> Dict[str, float]:
    if not dists:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "percentiles": {"25": 0.0, "50": 0.0, "75": 0.0, "90": 0.0},
            "N": 0,
        }
    arr = np.array(dists, float)
    return {
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "percentiles": {
            "25": float(np.percentile(arr, 25)),
            "50": float(np.percentile(arr, 50)),
            "75": float(np.percentile(arr, 75)),
            "90": float(np.percentile(arr, 90)),
        },
        "N": int(arr.size),
    }


def evaluate_metrics(
    gt_by_t_merged: Dict[int, List[dict]],
    pred_df: pd.DataFrame,
    thresholds: List[float] = None,
) -> Dict:
    if thresholds is None:
        thresholds = PREC_THRESH_M
    results = {
        "overall": {"detection": {}, "position": {}},
        "by_class": {
            CLASS_ID2NAME[c]: {"detection": {}, "position": {}} for c in CLASS_ID2NAME
        },
        "params": {
            "FRAME_SCALE": FRAME_SCALE,
            "FRAME_OFFSET": FRAME_OFFSET,
            "PREC_THRESH_M": thresholds,
            "MATCH_GATE_M": MATCH_GATE_M,
            "MATCH_GATE_BY_CLASS": MATCH_GATE_BY_CLASS,
            "GT_MERGE_RADIUS_M": GT_MERGE_RADIUS_M,
            "MERGE_RADIUS_BY_CLASS": MERGE_RADIUS_BY_CLASS,
            "GT_CLASS_MAP": "COCO→{0:ball,1:player,2:referee}",
        },
    }
    glob_tp = {thr: 0 for thr in thresholds}
    glob_fp = {thr: 0 for thr in thresholds}
    glob_fn = {thr: 0 for thr in thresholds}
    glob_dists = []
    cls_tp = {c: {thr: 0 for thr in thresholds} for c in CLASS_ID2NAME}
    cls_fp = {c: {thr: 0 for thr in thresholds} for c in CLASS_ID2NAME}
    cls_fn = {c: {thr: 0 for thr in thresholds} for c in CLASS_ID2NAME}
    cls_dists = {c: [] for c in CLASS_ID2NAME}

    frames = sorted(set(gt_by_t_merged.keys()) & set(pred_df["t"].unique()))
    thr_pos = max(thresholds)
    for t in frames:
        gt_anns = gt_by_t_merged.get(t, [])
        gt_xy_by_cls = defaultdict(list)
        for a in gt_anns:
            gt_xy_by_cls[a["class_id"]].append((a["field_x"], a["field_y"]))
        df_t = pred_df[pred_df["t"] == t]
        pr_xy_by_cls = defaultdict(list)
        for row in df_t.itertuples(index=False):
            pr_xy_by_cls[int(row.cls_id)].append((float(row.x), float(row.y)))
        for cls_id in CLASS_ID2NAME:
            G = gt_xy_by_cls.get(cls_id, [])
            P = pr_xy_by_cls.get(cls_id, [])
            nG, nP = len(G), len(P)
            if nG == 0 and nP == 0:
                continue
            D = None
            if nG and nP:
                D = np.zeros((nG, nP), dtype=float)
                for i, (Xg, Yg) in enumerate(G):
                    for j, (Xp, Yp) in enumerate(P):
                        D[i, j] = math.hypot(Xp - Xg, Yp - Yg)
            # posizione con soglia ampia clampata dal gate
            if nG and nP:
                gate_cls = MATCH_GATE_BY_CLASS.get(cls_id, MATCH_GATE_M)
                thr_eff = min(thr_pos, gate_cls)
                Cpos = np.where(D <= thr_eff, D, 1e6)
                rpos, cpos = linear_sum_assignment(Cpos)
                dists_pos = [
                    float(Cpos[i, j]) for i, j in zip(rpos, cpos) if Cpos[i, j] < 1e6
                ]
                cls_dists[cls_id].extend(dists_pos)
                glob_dists.extend(dists_pos)
            # detection per soglia
            for thr in thresholds:
                if nG == 0 and nP == 0:
                    tp = fp = fn = 0
                elif nG == 0:
                    tp, fp, fn = 0, nP, 0
                elif nP == 0:
                    tp, fp, fn = 0, 0, nG
                else:
                    gate_cls = MATCH_GATE_BY_CLASS.get(cls_id, MATCH_GATE_M)
                    thr_eff = min(thr, gate_cls)
                    C = np.where(D <= thr_eff, D, 1e6)
                    r, c = linear_sum_assignment(C)
                    tp = sum(1 for i, j in zip(r, c) if C[i, j] < 1e6)
                    fp = max(0, nP - tp)
                    fn = max(0, nG - tp)
                cls_tp[cls_id][thr] += tp
                cls_fp[cls_id][thr] += fp
                cls_fn[cls_id][thr] += fn
                glob_tp[thr] += tp
                glob_fp[thr] += fp
                glob_fn[thr] += fn

    for cls_id, cls_name in CLASS_ID2NAME.items():
        det = {}
        for thr in thresholds:
            det[f"threshold_{thr}m"] = calc_detection_metrics(
                cls_tp[cls_id][thr], cls_fp[cls_id][thr], cls_fn[cls_id][thr]
            )
        pos = calc_position_metrics(cls_dists[cls_id])
        results["by_class"][cls_name]["detection"] = det
        results["by_class"][cls_name]["position"] = pos

    det_overall = {}
    for thr in thresholds:
        det_overall[f"threshold_{thr}m"] = calc_detection_metrics(
            glob_tp[thr], glob_fp[thr], glob_fn[thr]
        )
    pos_overall = calc_position_metrics(glob_dists)
    results["overall"]["detection"] = det_overall
    results["overall"]["position"] = pos_overall
    results["frames_evaluated"] = len(frames)
    return results


# =========================
# METRICHE (tracking: CLEAR-MOT, IDF1)
# =========================
def build_gt_tracks_3d(gt_by_t_merged, link_gate_by_class=None, max_frame_gap=0):
    if link_gate_by_class is None:
        link_gate_by_class = {c: 1.0 for c in CLASS_ID2NAME}
    frames = sorted(gt_by_t_merged.keys())
    next_tid = 0
    last_by_cls = {c: [] for c in CLASS_ID2NAME}
    for idx, t in enumerate(frames):
        anns = gt_by_t_merged[t]
        by_cls = defaultdict(list)
        for a in anns:
            by_cls[a["class_id"]].append(a)
        for cls_id in CLASS_ID2NAME:
            cur = by_cls.get(cls_id, [])
            prev_list = last_by_cls.get(cls_id, [])
            if cur and prev_list:
                P = np.array([[px, py] for (px, py, _) in prev_list], float)
                C = np.array([[a["field_x"], a["field_y"]] for a in cur], float)
                D = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
                gate = link_gate_by_class.get(cls_id, 1.0)
                Cost = np.where(D <= gate, D, 1e6)
                r, c = linear_sum_assignment(Cost)
                assigned_cur = set()
                for i, j in zip(r, c):
                    if Cost[i, j] < 1e6:
                        tid = prev_list[i][2]
                        cur[j]["tid"] = tid
                        assigned_cur.add(j)
                for j, a in enumerate(cur):
                    if j not in assigned_cur:
                        a["tid"] = next_tid
                        next_tid += 1
            else:
                for a in cur:
                    a["tid"] = next_tid
                    next_tid += 1
            last_by_cls[cls_id] = [(a["field_x"], a["field_y"], a["tid"]) for a in cur]
    return gt_by_t_merged


def evaluate_tracking_3d(
    gt_with_tid_by_t: Dict[int, List[dict]],
    pred_df: pd.DataFrame,
    match_gate_by_class: Dict[int, float] = None,
    pos_threshold: float = None,
) -> Dict[str, float]:
    if match_gate_by_class is None:
        match_gate_by_class = {c: MATCH_GATE_M for c in CLASS_ID2NAME}
    if pos_threshold is None:
        pos_threshold = max(PREC_THRESH_M)
    total_gt = total_pred = total_matched = 0
    total_fp = total_fn = 0
    total_idsw = 0
    dists_all = []
    last_pred_for_gt: Dict[int, int] = {}
    matched_gtids_by_t: Dict[int, set] = defaultdict(set)
    presence_gtids_by_t: Dict[int, set] = defaultdict(set)
    frames = sorted(set(gt_with_tid_by_t.keys()) & set(pred_df["t"].unique()))
    for t in frames:
        gt_list = gt_with_tid_by_t.get(t, [])
        df_t = pred_df[pred_df["t"] == t]
        gt_by_cls = defaultdict(list)
        for g in gt_list:
            presence_gtids_by_t[t].add(g["tid"])
            gt_by_cls[g["class_id"]].append(g)
        pr_by_cls = defaultdict(list)
        for r in df_t.itertuples(index=False):
            pr_by_cls[int(r.cls_id)].append(r)
        for cls_id in CLASS_ID2NAME:
            G = gt_by_cls.get(cls_id, [])
            P = pr_by_cls.get(cls_id, [])
            nG, nP = len(G), len(P)
            total_gt += nG
            total_pred += nP
            if nG and nP:
                D = np.zeros((nG, nP), float)
                for i, g in enumerate(G):
                    gx, gy = float(g["field_x"]), float(g["field_y"])
                    for j, p in enumerate(P):
                        D[i, j] = math.hypot(float(p.x) - gx, float(p.y) - gy)
                gate = min(match_gate_by_class.get(cls_id, MATCH_GATE_M), pos_threshold)
                C = np.where(D <= gate, D, 1e6)
                r, c = linear_sum_assignment(C)
                m_this = 0
                for i, j in zip(r, c):
                    if C[i, j] >= 1e6:
                        continue
                    m_this += 1
                    dists_all.append(float(D[i, j]))
                    gt_tid = int(G[i]["tid"])
                    pred_id = int(P[j].track_id)
                    if (
                        gt_tid in last_pred_for_gt
                        and last_pred_for_gt[gt_tid] != pred_id
                    ):
                        total_idsw += 1
                    last_pred_for_gt[gt_tid] = pred_id
                    matched_gtids_by_t[t].add(gt_tid)
                total_matched += m_this
                total_fp += max(0, nP - m_this)
                total_fn += max(0, nG - m_this)
            elif nG == 0:
                total_fp += nP
            else:
                total_fn += nG
    # fragments
    total_frag = 0
    if presence_gtids_by_t:
        all_gtids = set().union(*presence_gtids_by_t.values())
        for tid in all_gtids:
            segments = 0
            inside = False
            for t in frames:
                present = tid in presence_gtids_by_t[t]
                matched = tid in matched_gtids_by_t[t]
                if present and matched and not inside:
                    segments += 1
                    inside = True
                elif (not matched or not present) and inside:
                    inside = False
            if segments > 0:
                total_frag += max(0, segments - 1)
    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(1, total_gt)
    motp_m = (sum(dists_all) / len(dists_all)) if dists_all else 0.0
    precision = total_matched / max(1, (total_matched + total_fp))
    recall = total_matched / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    return {
        "MOTA_3D": mota,
        "MOTP_3D_m": motp_m,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ID_Switches": total_idsw,
        "Fragments": total_frag,
        "FP": total_fp,
        "FN": total_fn,
        "GT_total": total_gt,
        "Matches": total_matched,
        "N_pos": len(dists_all),
    }


def evaluate_idf1_3d(gt_by_t_merged, pred_df, thresholds=None):
    thr_eval = max(thresholds) if thresholds else 1.0
    if not any("tid" in a for L in gt_by_t_merged.values() for a in L):
        _ = build_gt_tracks_3d(gt_by_t_merged)
    pair_counts = defaultdict(int)
    total_gt_det = 0
    total_pr_det = 0
    frames = sorted(set(gt_by_t_merged.keys()) & set(pred_df["t"].unique()))
    for t in frames:
        gt_list = gt_by_t_merged.get(t, [])
        df_t = pred_df[pred_df["t"] == t]
        gt_by_cls = defaultdict(list)
        for g in gt_list:
            gt_by_cls[g["class_id"]].append(g)
        pr_by_cls = defaultdict(list)
        for r in df_t.itertuples(index=False):
            pr_by_cls[int(r.cls_id)].append(r)
        for cls_id in CLASS_ID2NAME:
            G = gt_by_cls.get(cls_id, [])
            P = pr_by_cls.get(cls_id, [])
            nG, nP = len(G), len(P)
            total_gt_det += nG
            total_pr_det += nP
            if not (nG and nP):
                continue
            D = np.zeros((nG, nP), float)
            for i, g in enumerate(G):
                gx, gy = float(g["field_x"]), float(g["field_y"])
                for j, p in enumerate(P):
                    D[i, j] = math.hypot(float(p.x) - gx, float(p.y) - gy)
            gate = min(MATCH_GATE_BY_CLASS.get(cls_id, MATCH_GATE_M), thr_eval)
            C = np.where(D <= gate, D, 1e6)
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] >= 1e6:
                    continue
                gt_tid = int(G[i]["tid"])
                pr_id = int(P[j].track_id)
                pair_counts[(gt_tid, pr_id)] += 1
    if not pair_counts:
        return {
            "IDP": 0.0,
            "IDR": 0.0,
            "IDF1": 0.0,
            "IDTP": 0,
            "IDFP": total_pr_det,
            "IDFN": total_gt_det,
        }
    gt_tids = sorted({gt for gt, _ in pair_counts.keys()})
    pr_ids = sorted({pr for _, pr in pair_counts.keys()})
    idx_gt = {tid: i for i, tid in enumerate(gt_tids)}
    idx_pr = {pid: j for j, pid in enumerate(pr_ids)}
    M = np.zeros((len(gt_tids), len(pr_ids)), float)
    for (tid, pid), v in pair_counts.items():
        M[idx_gt[tid], idx_pr[pid]] = v
    r, c = linear_sum_assignment(-M)  # massimizza
    IDTP = int(sum(M[i, j] for i, j in zip(r, c)))
    IDFN = int(total_gt_det - IDTP)
    IDFP = int(total_pr_det - IDTP)
    IDP = IDTP / max(1, IDTP + IDFP)
    IDR = IDTP / max(1, IDTP + IDFN)
    IDF1 = 2 * IDP * IDR / max(1e-6, (IDP + IDR))
    return {
        "IDP": IDP,
        "IDR": IDR,
        "IDF1": IDF1,
        "IDTP": IDTP,
        "IDFP": IDFP,
        "IDFN": IDFN,
    }


# =========================
# PRETTY PRINT
# =========================
def _fmt_pct(x: float) -> str:
    return f"{100.0*x:5.1f}%"


def _fmt3(x: float) -> str:
    return f"{x:.3f}"


def _table(
    headers: List[str], rows: List[List[str]], title: Optional[str] = None
) -> str:
    cols = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            cols[i] = max(cols[i], len(str(c)))

    def fmt_row(r):
        return "│ " + " │ ".join(str(c).ljust(cols[i]) for i, c in enumerate(r)) + " │"

    top = "┌" + "┬".join("─" * (w + 2) for w in cols) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in cols) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in cols) + "┘"
    lines = [top, fmt_row(headers), mid]
    lines += [fmt_row(r) for r in rows] or [fmt_row(["—"] * len(headers))]
    lines.append(bot)
    if title:
        lines = [title] + lines
    return "\n".join(lines)


def print_summary_pretty(
    res: Dict, trk: Optional[Dict] = None, idf: Optional[Dict] = None
):
    print("\n" + "=" * 72)
    print("3D TRACKING EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Frames evaluated: {res.get('frames_evaluated', 0)}")

    # posizione
    pos = res["overall"]["position"]
    rows = [
        ["MAE (m)", _fmt3(pos["mae"])],
        ["RMSE (m)", _fmt3(pos["rmse"])],
        ["P50 (m)", _fmt3(pos["percentiles"]["50"])],
        ["P90 (m)", _fmt3(pos["percentiles"]["90"])],
        ["N match", str(pos["N"])],
    ]
    print(_table(["Overall Position", ""], rows))

    # detection overall per soglia
    drows = []
    for thr, v in res["overall"]["detection"].items():
        drows.append(
            [
                thr.replace("threshold_", "@").replace("m", " m"),
                _fmt_pct(v["precision"]),
                _fmt_pct(v["recall"]),
                _fmt_pct(v["f1"]),
                f"TP={v['tp']}",
                f"FP={v['fp']}",
                f"FN={v['fn']}",
            ]
        )
    print(
        _table(
            ["Det threshold", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            drows,
            title="\nDetection (overall)",
        )
    )

    # per-class
    grows = []
    for cname in ["ball", "player", "referee"]:
        blk = res["by_class"][cname]
        p = blk["position"]
        d = blk["detection"][
            f"threshold_{PREC_THRESH_M[min(1, len(PREC_THRESH_M)-1)]}m"
        ]  # tipicamente 0.5
        grows.append(
            [
                cname,
                _fmt3(p["mae"]),
                _fmt3(p["rmse"]),
                _fmt3(p["percentiles"]["50"]),
                str(p["N"]),
                _fmt_pct(d["precision"]),
                _fmt_pct(d["recall"]),
                _fmt_pct(d["f1"]),
            ]
        )
    print(
        _table(
            ["Class", "MAE", "RMSE", "P50", "N", "P@0.5", "R@0.5", "F1@0.5"],
            grows,
            title="\nPer-class",
        )
    )

    # tracking panels
    if trk:
        trows = [
            [
                _fmt3(trk["MOTA_3D"]),
                _fmt3(trk["MOTP_3D_m"]),
                _fmt_pct(trk["Precision"]),
                _fmt_pct(trk["Recall"]),
                _fmt_pct(trk["F1"]),
                str(trk["ID_Switches"]),
                str(trk["Fragments"]),
                str(trk["FP"]),
                str(trk["FN"]),
                str(trk["GT_total"]),
                str(trk["Matches"]),
                str(trk["N_pos"]),
            ]
        ]
        print(
            _table(
                [
                    "MOTA_3D",
                    "MOTP_3D(m)",
                    "Precision",
                    "Recall",
                    "F1",
                    "IDSW",
                    "Frag",
                    "FP",
                    "FN",
                    "GT",
                    "Matches",
                    "N_pos",
                ],
                trows,
                title="\nTracking 3D (CLEAR-MOT on field XY)",
            )
        )
    if idf:
        irows = [
            [
                _fmt_pct(idf["IDP"]),
                _fmt_pct(idf["IDR"]),
                _fmt_pct(idf["IDF1"]),
                str(idf["IDTP"]),
                str(idf["IDFP"]),
                str(idf["IDFN"]),
            ]
        ]
        print(
            _table(
                ["IDP", "IDR", "IDF1", "IDTP", "IDFP", "IDFN"],
                irows,
                title="\nIDF1 (global identity)",
            )
        )


# =========================
# OUTPUT
# =========================
def save_results(
    res: Dict, out_path: str, trk: Optional[Dict] = None, idf: Optional[Dict] = None
):
    def _conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, list):
            return [_conv(x) for x in o]
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        return o

    payload = {"metrics": res, "tracking": trk, "idf1": idf}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_conv(payload), f, indent=2)
    print(f"\nSaved metrics to: {out_path}")


# =========================
# MAIN
# =========================
def main():
    print("Starting 3D Tracking Evaluation")
    # 1) Predizioni
    pred_df = load_predictions(TRACKS3D_JSON)
    print("Pred class value counts:", pred_df["class"].value_counts().to_dict())
    print("Mapped cls_id counts:", pred_df["cls_id"].value_counts().to_dict())
    frames_pred = set(int(t) for t in pred_df["t"].unique())
    if not frames_pred:
        print("No valid prediction frames found. Check TRACKS3D_CSV.")
        return

    # 2) GT e calibrazioni
    gt_by_fc = load_gt_coco(COCO_GT_PATH)
    print(f"Loaded {len(gt_by_fc)} GT annotations from {COCO_GT_PATH}")
    cam_sizes = _collect_cam_image_sizes_from_coco(COCO_GT_PATH)
    calib = load_calibrations_and_H(
        CAMERA_DATA, cam_image_sizes=cam_sizes, alpha=ALPHA_RECT
    )
    print(f"Loaded cameras: {sorted(calib.keys())}")

    # 3) Allineamento, proiezione e dedup
    gt_aligned = align_gt_frames(gt_by_fc, frames_pred)
    print(f"Aligned {len(gt_aligned)} GT frames to prediction frames.")
    if not gt_aligned:
        print("No aligned frames. Check FRAME_SCALE/OFFSET.")
        return
    gt_projected = project_gt_to_field(gt_aligned, calib)
    print(
        f"Projected GT to field coordinates: {len(gt_projected)} frames with annotations."
    )
    gt_merged = dedup_gt_per_frame(gt_projected, merge_radius_m=GT_MERGE_RADIUS_M)
    total_gt = sum(len(v) for v in gt_merged.values())
    print(
        f"Deduplicated GT annotations: {total_gt} total across {len(gt_merged)} frames."
    )

    # 4) Post-processing delle predizioni
    # log base
    _log_tracks_basic(pred_df)
    # stitching (più prudente)
    pred_df = stitch_tracks(pred_df)
    # filtro lunghezza minima
    pred_df = filter_min_length(pred_df, MIN_TRACK_LEN_FRAMES)
    # filtro velocità "two-strike"
    pred_df = speed_filter_two_strike(pred_df)
    # smoothing zero-lag
    pred_df = smooth_tracks_zero_lag(pred_df, ZL_WINDOW)
    # NMS metrico framewise
    pred_df = metric_nms_framewise(pred_df, PRED_NMS_RADIUS_BY_CLS)

    # 5) Analisi lunghezze e coverage venendo incontro al log che ti piace
    spans = _track_spans(pred_df)
    lens = pd.Series({tid: L for tid, (_, _, L) in spans.items()})
    print(
        f"Pred tracks: count={len(lens)}, median_len={float(lens.median()):.1f} frames, P90={float(lens.quantile(0.9))}"
    )

    # 6) (opzionale) matched frames per track id: contiamo i match @max soglia
    from collections import defaultdict

    matched_by_track = defaultdict(int)
    used_frames = sorted(set(gt_merged.keys()) & set(pred_df["t"].unique()))
    thr = max(PREC_THRESH_M)
    for t in used_frames:
        df_t = pred_df[pred_df["t"] == t]
        G = gt_merged[t]
        for cls_id in CLASS_ID2NAME:
            g = [(a["field_x"], a["field_y"]) for a in G if a["class_id"] == cls_id]
            p = [
                (float(r.x), float(r.y), int(r.track_id))
                for r in df_t.itertuples(index=False)
                if int(r.cls_id) == cls_id
            ]
            if not (g and p):
                continue
            D = np.zeros((len(g), len(p)))
            for i, (gx, gy) in enumerate(g):
                for j, (px, py, _) in enumerate(p):
                    D[i, j] = math.hypot(px - gx, py - gy)
            C = np.where(
                D <= min(thr, MATCH_GATE_BY_CLASS.get(cls_id, MATCH_GATE_M)), D, 1e6
            )
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] < 1e6:
                    matched_by_track[p[j][2]] += 1
    if matched_by_track:
        m = pd.Series(matched_by_track)
        print(
            f"Matched frames per pred track: median={int(m.median())}, P90={int(m.quantile(0.9))}, max={int(m.max())}"
        )

    # 7) Valutazione
    used_frames = sorted(gt_merged.keys())
    pred_used = pred_df[pred_df["t"].isin(used_frames)].copy()
    res = evaluate_metrics(gt_merged, pred_used, thresholds=PREC_THRESH_M)

    # 8) Tracking (CLEAR-MOT) e IDF1
    trk = None
    idf = None
    if ENABLE_TRACKING_METRICS:
        gt_with_tid = build_gt_tracks_3d(
            gt_merged,
            link_gate_by_class=LINK_GATE_BY_CLASS,
            max_frame_gap=LINK_MAX_FRAME_GAP,
        )
        trk = evaluate_tracking_3d(
            gt_with_tid,
            pred_used,
            match_gate_by_class=MATCH_GATE_BY_CLASS,
            pos_threshold=max(PREC_THRESH_M),
        )
        idf = evaluate_idf1_3d(gt_with_tid, pred_used, thresholds=PREC_THRESH_M)

    # 9) Stampa “bella” e salvataggio
    print_summary_pretty(res, trk, idf)
    save_results(res, OUT_JSON, trk=trk, idf=idf)

    # 10) Coverage tracce GT (debug utile)
    if ENABLE_TRACKING_METRICS:
        coverage = defaultdict(int)
        presence = defaultdict(int)
        for t, lst in gt_with_tid.items():
            tids_present = {a["tid"] for a in lst}
            for tid in tids_present:
                presence[tid] += 1
            matched = set()
            df_t = pred_used[pred_used["t"] == t]
            for cls_id in CLASS_ID2NAME:
                G = [a for a in lst if a["class_id"] == cls_id]
                P = [r for r in df_t.itertuples(index=False) if int(r.cls_id) == cls_id]
                if not (G and P):
                    continue
                D = np.zeros((len(G), len(P)), float)
                for i, g in enumerate(G):
                    for j, p in enumerate(P):
                        D[i, j] = math.hypot(
                            float(p.x) - g["field_x"], float(p.y) - g["field_y"]
                        )
                thr = min(
                    MATCH_GATE_BY_CLASS.get(cls_id, MATCH_GATE_M), max(PREC_THRESH_M)
                )
                C = np.where(D <= thr, D, 1e6)
                r, c = linear_sum_assignment(C)
                for i, j in zip(r, c):
                    if C[i, j] < 1e6:
                        matched.add(G[i]["tid"])
            for tid in matched:
                coverage[tid] += 1
        if presence:
            cov_vals = [coverage[tid] / max(1, presence[tid]) for tid in presence]
            print(
                f"\nGT track coverage: median={np.median(cov_vals):.2f}, P10={np.percentile(cov_vals,10):.2f}, P90={np.percentile(cov_vals,90):.2f}"
            )


if __name__ == "__main__":
    main()
