# 3D Tracking Module

- Multi-view triangulation, global tracking and playback utilities for basketball player/referee analysis.
- Scripts expect rectified camera detections and calibration JSON in `../camera_data`.
- Outputs are written under `output/` (generated automatically when running the pipelines).

## Python Files
- `triangulation.py` – Loads per-camera tracking JSON, triangulates 3D points (with adaptive epipolar gating), refines them, and feeds a Kalman-based global tracker to produce `output/tracks_3d.csv` plus adaptive logs.
- `tracking.py` – Consumes `tracks_3d.csv`, runs an enhanced MOT-3D data-association/Kalman pipeline, and writes multi-object tracks with histories to `output/tracking3d_output_multi.json`.
- `playback_2d_tracking.py` – Matplotlib viewer for `tracking3d_output_multi.json`; projects tracked entities onto the XY plane with playback controls and display caps (12 players, 2 referees).
- `rectified_video.py` – Batch undistorts/rectifies raw videos in `../data/video/` using per-camera calibration and saves them to `../data/rectified/`.
- `__pycache__/` – Auto-generated Python cache directory (safe to ignore).

## Prerequisites
- Python 3.9+ with packages: `numpy`, `pandas`, `scipy`, `opencv-python`, `matplotlib` (see root `requirements.txt`).
- Calibration JSON and per-camera tracking JSONs in the expected relative locations (`../camera_data/cam_{id}/calib`, `output/tracking_results_rect_out*.json`, etc.).

## Typical Workflow
- Generate 3D points and global tracks:
  ```bash
  cd basketball_tracking/3DTracking
  python triangulation.py        # writes output/tracks_3d.csv and adaptive logs
  python tracking.py             # converts CSV to multi-target JSON tracks
  ```
- Inspect tracks in 2D:
  ```bash
  python playback_2d_tracking.py  # opens interactive viewer on output/tracking3d_output_multi.json
  ```
- Rectify raw camera videos (optional utility):
  ```bash
  python rectified_video.py       # processes ../data/video/out*.mp4 into ../data/rectified/
  ```

## Notes
- Adjust constants (e.g., file paths, gating thresholds, display limits) directly inside each script as needed.
- The pipelines assume 25 FPS; change `FPS` in `triangulation.py` and `tracking.py` together if your data uses a different frame rate.
- Output artifacts (CSV/JSON/plots) are overwritten on each run; archive them beforehand when necessary.
