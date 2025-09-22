import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INPUT_CSV = os.path.join(OUTPUT_DIR, "triangulated_3d.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "3d_tracks.csv")

# ---------------- Kalman Filter ----------------

@dataclass
class KalmanCV:
    dt: float = 1.0
    x: np.ndarray = field(default_factory=lambda: np.zeros((6,1), dtype=float))
    P: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=float) * 1e3)
    F: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    q_pos: float = 50.0
    q_vel: float = 200.0
    r_meas: float = 400.0

    def __post_init__(self):
        self.F = np.eye(6, dtype=float)
        self.F[0,3] = self.dt
        self.F[1,4] = self.dt
        self.F[2,5] = self.dt
        self.H = np.zeros((3,6), dtype=float)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.Q = np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel]).astype(float)
        self.R = np.eye(3, dtype=float) * self.r_meas

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def maha(self, z: np.ndarray) -> float:
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

# ------------- Track Management -------------

@dataclass
class Track:
    track_id: int
    kf: KalmanCV
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    last_frame: int = -1

class MultiClassTracker3D:
    def __init__(self, dt: float = 1.0, gate_thr: float = 16.0, max_misses: int = 15, min_hits: int = 3):
        self.dt = dt
        self.gate_thr = gate_thr
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks_by_class: Dict[str, List[Track]] = {"player": [], "referee": [], "ball": []}

    def _new_kf(self, z: np.ndarray) -> KalmanCV:
        kf = KalmanCV(dt=self.dt)
        kf.x[:3,0] = z.ravel()
        kf.P = np.diag([1e2,1e2,1e2, 1e4,1e4,1e4]).astype(float)
        return kf

    def _predict_all(self):
        for tracks in self.tracks_by_class.values():
            for t in tracks:
                t.kf.predict()
                t.misses += 1  # reset on successful update

    def _associate(self, tracks: List[Track], detections: np.ndarray) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        if len(tracks) == 0 or detections.size == 0:
            return [], list(range(len(tracks))), list(range(detections.shape[0]))
        cost = np.full((len(tracks), detections.shape[0]), 1e6, dtype=float)
        for i, t in enumerate(tracks):
            for j in range(detections.shape[0]):
                z = detections[j].reshape(3,1)
                d2 = t.kf.maha(z)
                if d2 <= self.gate_thr:
                    cost[i,j] = d2
        row, col = linear_sum_assignment(cost)
        matches, um_tracks, um_dets = [], [], []
        assigned_tracks, assigned_dets = set(), set()
        for r, c in zip(row, col):
            if cost[r,c] >= 1e6:
                continue
            matches.append((r,c))
            assigned_tracks.add(r)
            assigned_dets.add(c)
        for i in range(len(tracks)):
            if i not in assigned_tracks:
                um_tracks.append(i)
        for j in range(detections.shape[0]):
            if j not in assigned_dets:
                um_dets.append(j)
        return matches, um_tracks, um_dets

    def step(self, frame_idx: int, dets_by_class: Dict[str, np.ndarray]) -> List[Dict]:
        self._predict_all()
        outputs = []

        for cls in self.tracks_by_class.keys():
            tracks = self.tracks_by_class[cls]
            dets = dets_by_class.get(cls, np.zeros((0,3), dtype=float))

            matches, um_tracks, um_dets = self._associate(tracks, dets)

            # Matched updates
            for ti, dj in matches:
                t = tracks[ti]
                z = dets[dj].reshape(3,1)
                t.kf.update(z)
                t.hits += 1
                t.misses = 0
                t.last_frame = frame_idx
                if not t.confirmed and t.hits >= self.min_hits:
                    t.confirmed = True

            # New tracks
            for dj in um_dets:
                z = dets[dj].reshape(3,1)
                kf = self._new_kf(z)
                tr = Track(track_id=self.next_id, kf=kf, hits=1, misses=0, confirmed=(self.min_hits<=1), last_frame=frame_idx)
                self.next_id += 1
                tracks.append(tr)

            # Prune dead
            kept = []
            for t in tracks:
                if t.misses <= self.max_misses:
                    kept.append(t)
            self.tracks_by_class[cls] = kept

            # Emit confirmed tracks
            for t in self.tracks_by_class[cls]:
                if t.confirmed:
                    x = t.kf.x.ravel()
                    outputs.append({
                        "frame": frame_idx,
                        "class": cls,
                        "track_id": t.track_id,
                        "X": float(x[0]),
                        "Y": float(x[1]),
                        "Z": float(x[2]),
                        "VX": float(x[3]),
                        "VY": float(x[4]),
                        "VZ": float(x[5]),
                    })
        return outputs

# ------------- IO -------------

def load_triangulated_points(path: str) -> Dict[int, Dict[str, List[List[float]]]]:
    """
    Reads triangulated_3d.csv with header:
    frame,class,track_id_cam2,track_id_cam4,track_id_cam13,X,Y,Z,reproj_err_px
    Returns dict: frame -> class -> list of [X,Y,Z]
    """
    by_frame: Dict[int, Dict[str, List[List[float]]]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fi = int(row["frame"])
            except:
                continue
            cls = row["class"]
            if cls not in {"player", "referee", "ball"}:
                continue
            x = float(row["X"]); y = float(row["Y"]); z = float(row["Z"])
            by_frame.setdefault(fi, {}).setdefault(cls, []).append([x,y,z])
    return by_frame

def write_tracks(rows: List[Dict], path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","class","track_id","X","Y","Z","VX","VY","VZ"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ------------- Main -------------

def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}")
    data = load_triangulated_points(INPUT_CSV)

    # Determine sorted frame list
    frames = sorted(data.keys())

    # If FPS known, set dt = 1/FPS here; default dt=1 frame
    tracker = MultiClassTracker3D(dt=1.0, gate_thr=16.0, max_misses=15, min_hits=3)

    all_rows = []
    for fidx in frames:
        dets_by_class = {}
        for cls in {"player", "referee", "ball"}:
            pts = data.get(fidx, {}).get(cls, [])
            if len(pts) > 0:
                dets_by_class[cls] = np.asarray(pts, dtype=float)
            else:
                dets_by_class[cls] = np.zeros((0,3), dtype=float)
        outs = tracker.step(fidx, dets_by_class)
        all_rows.extend(outs)

    write_tracks(all_rows, OUTPUT_CSV)
    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
