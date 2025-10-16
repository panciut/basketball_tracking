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
    
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), alpha=0, newImgSize=(width, height)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        mtx, dist, None, new_mtx, (width, height), cv2.CV_32FC1
    )

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
    directory = os.path.dirname(os.path.abspath(__file__))  # absolute base dir of script
    input_dir = os.path.normpath(os.path.join(directory, "..", "data/video"))
    output_dir = os.path.normpath(os.path.join(directory, "..", "data/rectified"))

    videos = glob.glob(os.path.join(input_dir, "out*.mp4"))
    if not videos:
        print(f"No videos found in: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for video_path in videos:
        basename = os.path.basename(video_path)
        match = re.search(r'out(\d+)\.mp4', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join(directory, "..", f"camera_data/cam_{cam_index}/calib/camera_calib.json")  # absolute
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
