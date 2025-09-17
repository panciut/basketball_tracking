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

CAMERA_DATA_FILES = "../camera_data"


def load_calibration():
    root = Path(CAMERA_DATA_FILES)
    camera_directories = [d for d in root.glob("cam_*") if (d / "calib" / "camera_calib.json").exists()]

def main():
    load_calibration()


if __name__ == "__main__":
    main()
