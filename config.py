# config.py
VIDEO = "out13"
VIDEO_PATH = "data/video/"+VIDEO+".mp4"
MODEL_PATH = "yolov8x.pt"
OUTPUT_PATH = "output/output"+VIDEO+".mp4"
DETECTIONS_PATH = "output/detections.json"
TRACKING_JSON_PATH = f"output/tracking_results_{VIDEO}.json"


MASKS_DIR = "output/foreground_masks"
OUTPUT_FILTERED_PATH = "output/detections_filtered.json"
OUTPUT_FILTERED_VIDEO = "output/output_filtered.mp4"
FG_THRESHOLD = 0.10

ANNOT_PATH = "data/annotations/out13_frame_0001_png.rf.7c88fdea1b9bcb932738d79c70a12539.jpg"
COCO_PATH = "data/annotations/_annotations.coco.json"
OUTPUT_TARGETS_PATH = "output/first_frame_targets.json"

RUN_YOLO_DETECTION = False
RUN_FOREGROUND_EXTRACTION = True
RUN_FILTERING = True
RUN_LABEL_ASSIGNMENT = True
RUN_TRACKING = True

IMG_SIZE = 1280
ALLOWED_CLASSES = ["person", "sports ball"]
LOG_INTERVAL = 30
HISTORY = 250
VAR_THRESHOLD = 12
DETECT_SHADOWS = True
MIN_CONTOUR_AREA = 500
IOU_THRESHOLD = 0.5
MAX_SEARCH_FRAMES = 4