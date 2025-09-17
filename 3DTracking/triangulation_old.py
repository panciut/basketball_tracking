import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json

# Load camera parameters from JSON files
# Each file should contain 'mtx', 'dist', 'rvecs', and 'tvecs' keys
# return a list of dictionaries with camera parameters
def load_camera_params_from_files(cam_files):
    cameras = []
    for file_path in cam_files:
        with open(file_path, 'r') as f:
            cam_data = json.load(f)
        
        cam = {
            "K": np.array(cam_data["mtx"]),
            "dist": np.array(cam_data["dist"][0]),  # unwrap 2D list
            "rvec": np.array([v[0] for v in cam_data["rvecs"]]),  # flatten
            "tvec": np.array([v[0] for v in cam_data["tvecs"]])
        }
        cameras.append(cam)

    return cameras


def build_projection_matrices(cams):
    projections = []
    for cam in cams:
        R, _ = cv2.Rodrigues(cam["rvec"])
        T = cam["tvec"].reshape(3, 1)
        P = cam["K"] @ np.hstack((R, T))
        projections.append(P)
    return projections


def triangulate_n_views(pts, Projections):
    A = []
    for pt, P in zip(pts, Projections):
        x, y = pt[0], pt[1]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]

# ----------- DUMMY TRACKING DATA (YOU SHOULD REPLACE THIS) -----------
tracking_data = {
    0: {1: [(500, 400), (520, 410), (505, 395)],
        2: [(600, 420), (615, 430), (602, 418)]},
    1: {1: [(510, 405), (530, 415), (515, 400)],
        2: [(610, 425), (625, 435), (612, 420)]},
    2: {1: [(520, 410), (540, 420), (525, 405)],
        2: [(620, 430), (635, 440), (622, 422)]}
}
# Format: {frame: {track_id: [(x1, y1), (x2, y2), (x3, y3)]}}
with open("output/tracking_results_out2.json") as f2:
    tracking_cam2 = json.load(f2)
with open("output/tracking_results_out4.json") as f4:
    tracking_cam4 = json.load(f4)
with open("output/tracking_results_out13.json") as f13:
    tracking_cam13 = json.load(f13)

frame_keys = set(tracking_cam2.keys()) & set(tracking_cam4.keys()) & set(tracking_cam13.keys())
tracking_data = {}

for key in sorted(frame_keys, key=int):
    i = int(key)
    tracking_data[i] = {}
    for det2 in tracking_cam2[key]:
        player_id = str(det2["id"])
        # Find corresponding detections in other cameras by player_id
        det4 = next((d for d in tracking_cam4[key] if str(d["id"]) == player_id), None)
        det13 = next((d for d in tracking_cam13[key] if str(d["id"]) == player_id), None)
        if det4 is not None and det13 is not None:
            tracking_data[i][player_id] = [
                det2["bbox"][:2],  # or whatever 2D point you want to use
                det4["bbox"][:2],
                det13["bbox"][:2]
            ]

# ----------- TRIANGULATE ALL OBJECTS OVER TIME -----------
#camera_data/cam_2/calib/camera_calib_real.json
#camera_data/cam_4/calib/camera_calib_real.json
#camera_data/cam_13/calib/camera_calib_real.json

cam_files = ["camera_data/cam_2/calib/camera_calib_real.json", "camera_data/cam_4/calib/camera_calib_real.json", "camera_data/cam_13/calib/camera_calib_real.json"]  # adjust paths as needed
cams = load_camera_params_from_files(cam_files)

projections = build_projection_matrices(cams)
tracks_3d = {}  # {track_id: [X1, X2, X3...]}

for frame_id in sorted(tracking_data.keys()):
    for track_id, pts_2d in tracking_data[frame_id].items():
        pts = [np.array(pt, dtype=np.float32) for pt in pts_2d]
        X = triangulate_n_views(pts, projections)

        if track_id not in tracks_3d:
            tracks_3d[track_id] = []
        tracks_3d[track_id].append(X)

# ----------- 3D VISUALIZATION WITH MATPLOTLIB -----------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green', 'orange', 'purple']

def update(frame_idx):
    ax.clear()
    ax.set_title(f"3D Tracking - Frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-20000, 20000)
    ax.set_zlim(-10000, 10000)

    for i, (track_id, traj) in enumerate(tracks_3d.items()):
        if frame_idx < len(traj):
            X = traj[frame_idx]
            ax.scatter(X[0], X[1], X[2], c=colors[i % len(colors)], label=f'ID {track_id}')
            ax.plot([p[0] for p in traj[:frame_idx+1]],
                    [p[1] for p in traj[:frame_idx+1]],
                    [p[2] for p in traj[:frame_idx+1]],
                    c=colors[i % len(colors)])
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(tracking_data), interval=1000)
plt.show()
# ani.save("3d_tracking_demo.mp4", writer="ffmpeg")  # optional: save to video
