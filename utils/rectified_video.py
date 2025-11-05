import cv2
import numpy as np
import json
import os
import glob
import re

INPUT_DIR = "../data/video"
OUTPUT_DIR = "../data/rectified"

def load_calibration(calib_path):
    with open(calib_path, "r") as f:
            data = json.load(f)
    mtx = np.array(data.get("K", data.get("mtx")), dtype=float)
    rvec = np.array(data.get("rvec", data.get("rvecs")), dtype=float).reshape(-1)[:3]
    tvec = np.array(data.get("tvec", data.get("tvecs")), dtype=float).reshape(-1)[:3]
    dist = np.array(data.get("dist", data.get("distCoeffs", [0,0,0,0,0])), dtype=float).reshape(-1)
    return mtx, rvec, tvec, dist

def process_video(video_path, calib_path, output_path, cam_index):
    mtx, rvec, tvec, dist = load_calibration(calib_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), alpha=0, newImgSize=(width, height)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        mtx, dist, None, new_mtx, (width, height), cv2.CV_32FC1
    )

    # once you have computed the new camera parameters for the undistortion
    # save the new calibration file as camera_calib_rectified.json
    rectified_calib = {
        "mtx": new_mtx.tolist(),
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],  # No distortion after rectification
        "tvec": tvec.tolist(),
        "rvec": rvec.tolist()
    }
    rectified_calib_path = f"../camera_data/cam_{cam_index}/calib/cam_{cam_index}_calib_rectified.json"
    with open(rectified_calib_path, "w") as f:
        json.dump(rectified_calib, f, indent=4)
    print(f"Saved rectified calibration to: {rectified_calib_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rectified_frame = cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        out.write(rectified_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames for {video_path}")

    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

def main():

    videos = glob.glob(os.path.join(INPUT_DIR, "out*.mp4"))
    if not videos:
        print(f"No videos found in: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for video_path in videos:
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            calib_path = f"../camera_data/cam_{cam_index}/calib/camera_calib.json"  # absolute
            if not os.path.exists(calib_path):
                print(f"[WARNING] Calibration file does not exist: {calib_path} -- skipping this video.")
                continue
        else:
            print("Could not extract camera index from filename:", video_path)
            continue
        output_path = os.path.join(OUTPUT_DIR, basename)
        print(f"Processing {video_path} using calibration file {calib_path}...")
        process_video(video_path, calib_path, output_path, cam_index)

if __name__ == "__main__":
    main()
