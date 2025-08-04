import cv2
import numpy as np
import json
import os
import glob
import re

def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def process_video(video_path, calib_path, output_path):
    mtx, dist = load_calibration(calib_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    undistorted_map = undistorted_pts.reshape(height, width, 2)
    map_x = undistorted_map[:, :, 0]
    map_y = undistorted_map[:, :, 1]
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rectified_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        out.write(rectified_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames for {video_path}")
    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

def main():
    video_files = glob.glob("data/video/out*.mp4")  # Find all outXX.mp4 videos in the data/video directory
    output_dir = "output/rectified"  # folder path where to save the rectified videos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_path in video_files:
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join("data", "camera_data", f"cam_{cam_index}", "calib", "camera_calib.json")
            if not os.path.exists(calib_path):
                print(f"[WARNING] Calibration file does not exist: {calib_path} -- skipping this video.")
                continue
        else:
            print("Could not extract camera index from filename:", video_path)
            continue
        output_path = os.path.join(output_dir, basename)
        print(f"Processing {video_path} using calibration file {calib_path}...")
        process_video(video_path, calib_path, output_path)

if __name__ == "__main__":
    main()