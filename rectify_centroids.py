import os
import json
import glob
import re
import numpy as np
import cv2
import config

def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [cx, cy]

def load_calibration(calib_path):
    print(f"[INFO] Loading calibration: {calib_path}")
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    # Load extrinsics if present
    rvec = np.array(calib.get("rvecs", []), dtype=np.float32).reshape(-1, 1)
    tvec = np.array(calib.get("tvecs", []), dtype=np.float32).reshape(-1, 1)
    return mtx, dist, rvec, tvec

def compute_projection_matrix(mtx, rvec, tvec):
    if rvec.size == 0 or tvec.size == 0:
        print("[WARNING] No extrinsics in calibration file!")
        return None
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec))
    P = mtx @ Rt
    print("[INFO] Projection matrix P = K[R|t]:")
    print(P)
    return P

def undistort_centroid(centroid, mtx, dist):
    pt = np.array([[[centroid[0], centroid[1]]]], dtype=np.float32)
    undist = cv2.undistortPoints(pt, mtx, dist, P=mtx)
    cx_u, cy_u = undist[0, 0]
    return [float(cx_u), float(cy_u)]

def process_detection_file(detections_path, calib_path, output_path):
    print(f"[INFO] Processing detections: {detections_path}")
    mtx, dist, rvec, tvec = load_calibration(calib_path)
    P = compute_projection_matrix(mtx, rvec, tvec)
    with open(detections_path, "r") as f:
        detections_per_frame = json.load(f)

    centroids_per_frame = {}
    total_centroids = 0
    for frame_idx, detections in detections_per_frame.items():
        frame_centroids = []
        for det in detections:
            centroid = compute_centroid(det['bbox'])
            undist_centroid = undistort_centroid(centroid, mtx, dist)
            frame_centroids.append({
                "centroid": centroid,
                "undistorted_centroid": undist_centroid,
                "label": det["label"],
                "conf": det.get("conf", None),
                "bbox": det["bbox"]
            })
            total_centroids += 1
        centroids_per_frame[frame_idx] = frame_centroids
        if int(frame_idx) % 50 == 0:
            print(f"[DEBUG] Frame {frame_idx}: processed {len(frame_centroids)} centroids.")

    with open(output_path, "w") as f:
        json.dump(centroids_per_frame, f, indent=2)
    print(f"[SUCCESS] Centroid data saved to: {output_path} | Total centroids: {total_centroids}")
    proj_path = output_path.replace(config.CENTROIDS_PREFIX, config.PROJ_PREFIX).replace(".json", ".npy")
    if P is not None:
        np.save(proj_path, P)
        print(f"[INFO] Projection matrix saved to: {proj_path}")

def main():
    detection_files = glob.glob(config.DETECTIONS_GLOB)
    print(f"[INFO] Found {len(detection_files)} detection files.")
    if not detection_files:
        print("[WARNING] No detection files found. Exiting.")
        return

    for detections_path in detection_files:
        base = os.path.basename(detections_path)
        print(f"\n[INFO] Starting file: {base}")
        match = re.search(r'detections_out(\d+)', base)
        if not match:
            print(f"[WARNING] Could not extract camera index from filename: {base}, skipping.")
            continue
        cam_index = match.group(1)
        calib_path = os.path.join(config.CALIB_ROOT, f"cam_{cam_index}", "calib", config.CALIB_FILENAME)
        if not os.path.exists(calib_path):
            print(f"[WARNING] Calibration file does not exist: {calib_path} -- skipping {base}.")
            continue

        outname = base.replace("detections_", config.CENTROIDS_PREFIX)
        output_path = os.path.join(config.OUTPUT_DIR, outname)
        print(f"[INFO] Writing output to: {output_path}")
        process_detection_file(detections_path, calib_path, output_path)

    print("\n=== DONE ===")
    print("TODO: Handle synchronization between cameras if needed.")

if __name__ == "__main__":
    main()