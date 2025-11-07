# 3D Multi‑Object Tracking — README

This folder contains a **camera‑triangulation → 3D tracking → evaluation → visualization** pipeline tailored for team‑sports (basketball) in multi‑camera settings.

It is designed to be **modular, reproducible, and class‑aware** (player / referee / ball), with covariance‑aware triangulation, robust 3D tracking, and clear metrics.

---

## Contents

- **`triangulation.py`** — builds 3D points from per‑camera tracks using epipolar gating, N‑view clustering, robust refinement, and exports per‑point **3×3 covariance** in meters² to `output/triangulations_3d.csv`.
- **`tracker.py`** — runs a class‑aware 3D Kalman tracker from the triangulated CSV and writes `output/tracking3d_output.json`.
- **`metrics.py`** — evaluates predictions vs. COCO‑style GT (projected to field plane) and reports detection/position metrics plus **CLEAR‑MOT (3D) and IDF1**. Saves `output/metrics_summary.json`.
- **`visualize_tracks.py`** — interactive 2D viewer that shows **only “actual” player points** (non‑interpolated / visible) on the ground plane with play/pause, slider, and ±25‑frame steps.

---

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U numpy pandas scipy opencv-python matplotlib
```

Python ≥ 3.9 is recommended.

### 2) Expected inputs

- **Calibrations** (rectified intrinsics/extrinsics) for each camera, JSON with keys like `K / mtx`, `rvec / tvec`, and optionally `dist/distCoeffs`.
- **Per‑camera tracking outputs** (JSON). Flexible formats are supported (flat lists, dicts by frame, or COCO‑like). Each detection needs `frame`, `bbox` (or `tlwh/xywh`), optional `id/track_id`, `score`, and `label` (e.g., `"player"`, `"referee"`, `"ball"`).

### 3) Pipeline in one glance

```bash
# A) Triangulation (multi‑view → 3D CSV with covariance)
python triangulation.py

# B) 3D tracking (CSV → tracking3d_output.json)
python tracker.py

# C) Evaluation (JSON + GT + calibrations → metrics_summary.json)
python metrics.py

# D) Visualization (players on plane, “actual” only)
python visualize_tracks.py
```

---

## Module details & key settings

### A) Triangulation (`triangulation.py`)

**What it does**
- Loads rectified calibrations for cam_2 / cam_4 / cam_13.
- Loads per‑camera tracks (supports multiple JSON schemas).
- Builds **pairwise epipolar matches**, clusters detections across cameras (N‑view), then **triangulates and refines** each 3D point with a robust (Huber) least‑squares.
- Computes **per‑camera residuals**, **RMS reprojection**, and a **3×3 covariance** via Gauss–Newton (`(JᵀJ)⁻¹`) scaled and clamped. Coordinates are exported in **meters**.
- Uses **label‑aware image points**: *ball → bbox center*; *player/referee → footpoint* (bottom‑center) to stabilize on‑ground actors.
- Optional **adaptive controller** to tune the epipolar gate and per‑camera weights over time.

**Important knobs (edit at top of the file)**
- `CALIB*_PATH`, `TRACKS*_PATH`, `OUTPUT_CSV`
- Gating: `EPI_GATE_PX_INIT`, `MAX_RMS_REPROJ_PX`, `USE_EPIPOLAR`
- Label‑aware UVs: `uv_by_label()`; primary camera tie‑break for labels: `PRIMARY_CAM`
- Units: `WORLD_SCALE = 1/1000` (meters if `t` was in mm in calib)
- Covariance clamps: `COV_EIG_MIN/MAX`, damping `COV_DAMP`

**CSV schema (`output/triangulations_3d.csv`)**
```
frame,time,label,cam2_id,cam4_id,cam13_id,x,y,z,n_views,rms_reproj_px,
res_cam2_px,res_cam4_px,res_cam13_px,cov_xx,cov_xy,cov_xz,cov_yy,cov_yz,cov_zz
```
- `(x,y,z)` are meters (after `WORLD_SCALE`)
- `cov_*` is the symmetric 3×3 covariance packed as: xx, xy, xz, yy, yz, zz

---

### B) 3D Tracking (`tracker.py`)

**What it does**
- Reads the CSV (candidates: `triangulations_3d.csv` or `output/triangulations_3d.csv`).
- Per‑measurement **covariance‑aware** association with **Mahalanobis (3D)** cost (fast 2D pre‑gate for speed).
- Class‑aware lifecycle tuning (players/referees confirm after more hits, ball looser, gravity on Z for ball prediction).
- Quality filters: allow single‑view for ball; clamp unrealistic |Z| for non‑ball; uncertainty‑aware **dedup** + **cross‑class suppression** (player vs referee).
- Outputs `output/tracking3d_output.json` with, for each track: `{track_id, label, confidence, history:[{frame,t,x[3],interp,conf}]}`.

**Key knobs**
- Lifecycle: `CLASS_TUNING[min_hits_confirm, max_misses, max_speed_mps]`
- Gating: `CHI2_GATE_2DOF/3DOF`
- Process noise per class: `ACCEL_NOISE`; gravity for ball: `BALL_GRAVITY`
- Measurement covariance synthesis when missing: `SIGMA_POS_BASE/MIN/MAX`, `Z_INFLATION_BY_VIEWS`

---

### C) Evaluation (`metrics.py`)

**What it does**
- Loads predictions from `output/tracking3d_output.json` (or CSV) and **COCO GT** (rectified).
- Aligns GT time to predictions with `FRAME_SCALE` & `FRAME_OFFSET` (e.g., GT 5 fps → pred 25 fps).
- Projects GT to **field XY** via camera homographies, deduplicates multi‑view GT per frame/class.
- Post‑processes predictions (optional): **track stitching**, **min‑length filter**, **two‑strike speed filter**, **zero‑lag smoothing**, and **metric NMS** per frame/class.
- Computes:
  - **Detection/position** metrics per class & overall (precision/recall/F1 at multiple meter thresholds; MAE/RMSE/P50/P90 on matched distances).
  - **3D CLEAR‑MOT** (on field XY): `MOTA_3D`, `MOTP_3D(m)`, IDSW, Fragments, FP/FN.
  - **IDF1** (global identity).  
- Writes a pretty console summary and saves `output/metrics_summary.json`.

**Paths to set**
- `TRACKS3D_JSON`, `COCO_GT_PATH`, `CAMERA_DATA`, `OUT_JSON`

**Important knobs**
- Time mapping: `FRAME_SCALE`, `FRAME_OFFSET`
- Matching gates: `MATCH_GATE_M`, `MATCH_GATE_BY_CLASS`
- GT dedup radius per class: `MERGE_RADIUS_BY_CLASS`
- Post‑proc toggles: `ENABLE_STITCH`, `MIN_TRACK_LEN_FRAMES`, `ENABLE_SPEED_FILTER`, `ENABLE_SMOOTH_ZL`, `ENABLE_PRED_NMS`

---

### D) Visualization (`visualize_tracks.py`)

**What it does**
- Reads `output/tracking3d_output.json`, **filters to label == "player"** and only **“actual”** history points (not `interp`, not `lost`, `visible==True` if present).
- Interactive matplotlib viewer:
  - Space = Play/Pause
  - ←/→ = −25/+25 frames
  - Slider = random access

**Tip**: If bounds look off, check units in the pipeline and FPS/frame indices.

---

## Data conventions

- **Labels**: `"player"`, `"referee"`, `"ball"` are recognized across modules.
- **Ground plane**: metrics evaluate on **field XY**; Z is ignored except for QC.
- **Units**: world coordinates are meters (triangulation applies `WORLD_SCALE`). Ensure calibration translations are in the expected units (mm vs m).

---

## Reproducibility notes

- Determinism is limited by floating‑point and SVD/LS solvers, but the pipeline is stateless given fixed inputs.
- For consistent timings, fix `FPS` and ensure GT/pred time alignment (`FRAME_SCALE`, `FRAME_OFFSET`).

---

## Troubleshooting

- **`Input CSV not found` (tracker)** → Verify `triangulations_3d.csv` exists in the working dir or at `output/triangulations_3d.csv`.
- **`No aligned frames. Check FRAME_SCALE/OFFSET.` (metrics)** → Your GT fps or offset differs; adjust `FRAME_SCALE` and `FRAME_OFFSET`.
- **Empty/near‑empty tracks** → Lower `MIN_VIEWS_DEFAULT` (ball already allows 1), relax `MAX_RMS_REPROJ_PX`, or widen `EPI_GATE_PX_INIT`.
- **Players “flying” (|Z| too large)** → Make sure `WORLD_SCALE` matches calibration units; verify rectification and label‑aware UVs.
- **Many ID switches** → Increase `min_hits_confirm` for players/referees; check measurement covariance synthesis and speed limits.
- **Viewer shows nothing** → Ensure history entries aren’t all `interp=True` and labels are exactly `"player"`.

---

## Suggested folder layout

```
project/
├─ camera_data/
│  ├─ cam_2/calib/cam_2_calib_rectified.json
│  ├─ cam_4/calib/cam_4_calib_rectified.json
│  └─ cam_13/calib/cam_13_calib_rectified.json
├─ data/
│  └─ rectified/_annotations_rectified.coco.json
├─ output/
│  ├─ tracking_results_rect_out2.json
│  ├─ tracking_results_rect_out4.json
│  ├─ tracking_results_rect_out13.json
│  ├─ triangulations_3d.csv
│  ├─ tracking3d_output.json
│  └─ metrics_summary.json
├─ triangulation.py
├─ tracker.py
├─ metrics.py
└─ visualize_tracks.py
```

---

## Minimal examples

**Triangulation only**
```bash
python triangulation.py
head -n 5 output/triangulations_3d.csv
```

**Tracking only (from existing CSV)**
```bash
python tracker.py
jq '.[0]' output/tracking3d_output.json  # view first track (requires jq)
```

**Evaluation only**
```bash
python metrics.py  # prints tables and saves output/metrics_summary.json
```

**Viewer**
```bash
python visualize_tracks.py  # opens an interactive window
```

---

## Extending

- Add cameras: drop new rectified calibration JSONs and extend the loader similarly to cam_2/cam_4/cam_13.
- Change classes: adjust `cls_key()` and class‑specific thresholds/tuning in both triangulation and tracker.
- Replace per‑camera detector/tracker: keep the per‑frame JSON schema (frame, bbox, id, score, label) or extend `load_tracks_generic()`.

---

## License & citation

Add your project’s license and citation here (paper, dataset, etc.).

---

**Maintainers**: Progetto Computer Vision — 3D tracking module.
