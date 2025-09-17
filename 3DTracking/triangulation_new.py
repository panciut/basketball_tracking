import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from collections import Counter
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment

# Directory containing camera calibration data
# structurated as cam_01, cam_02, ... with inside calib/camera_calib.json and metadata.json
CAMERA_DATA_FILES = "../camera_data"


def load_calibration():
    root = Path(CAMERA_DATA_FILES)
    calib = {}
    camera_directories = [
        d for d in root.glob("cam_*") if (d / "calib" / "camera_calib.json").exists()
    ]
    if camera_directories:
        for cam_dir in camera_directories:
            with open(f"{cam_dir}/calib/camera_calib.json", "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            K = np.array(calib_data["mtx"], dtype=float).reshape(3, 3)
            R, _ = cv2.Rodrigues(np.array(calib_data["rvecs"], dtype=float))
            T = np.array(calib_data["tvecs"], dtype=float)
            with open(f"{cam_dir}/metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            Rt = np.hstack((R, T))
            P = K @ Rt

            w = metadata["imsize"][0]
            h = metadata["imsize"][1]
            calib[cam_dir.name] = {"K": K, "R": R, "t": T, "P": P, "image_size": (w, h)}
    return calib

def main():
    calib = load_calibration()
    print(calib)


if __name__ == "__main__":
    main()
