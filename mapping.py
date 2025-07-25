# main_assign_labels.py

import cv2
import json
import numpy as np
import os
from utils import find_exact_matching_frame

# --- CONFIGURATION ---
ANNOT_PATH = 'data/annotations/out13_frame_0001_png.rf.7c88fdea1b9bcb932738d79c70a12539.jpg'
COCO_PATH = 'data/annotations/_annotations.coco.json'
DETECTIONS_PATH = 'output/detections.json'
VIDEO_PATH = 'data/video/out13.mp4'
OUTPUT_TARGETS_PATH = 'output/first_frame_targets.json'
IOU_THRESHOLD = 0.5
MAX_SEARCH_FRAMES = 4  # Number of video frames to search for a match

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / (boxAArea + boxBArea - interArea)

# --- Load COCO annotation file ---
with open(COCO_PATH) as f:
    coco = json.load(f)

id2file = {img['id']: img['file_name'] for img in coco['images']}
cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

annot_filename = os.path.basename(ANNOT_PATH)
gt_bboxes = []
for ann in coco['annotations']:
    file = id2file[ann['image_id']]
    if file == annot_filename:
        label = cat_map[ann['category_id']]
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h]
        gt_bboxes.append({'bbox': bbox, 'label': label})

print(f"[INFO] Found {len(gt_bboxes)} annotations for {annot_filename}.")

# --- Find the matching frame in the video using utils.py ---
frame_num = find_exact_matching_frame(ANNOT_PATH, VIDEO_PATH, max_search=MAX_SEARCH_FRAMES)
print(f"[INFO] The first annotation frame corresponds to video frame {frame_num}.")

# --- Load YOLO detections ---
with open(DETECTIONS_PATH) as f:
    detections = json.load(f)
det_bboxes = detections.get(str(frame_num), [])

# --- Assign annotation labels to closest YOLO detection (by IoU) ---
used_detections = set()
for gt in gt_bboxes:
    best_iou = 0
    best_det_idx = None
    for idx, det in enumerate(det_bboxes):
        if idx in used_detections:
            continue
        iou_score = iou(gt['bbox'], det['bbox'])
        if iou_score > best_iou:
            best_iou = iou_score
            best_det_idx = idx
    if best_det_idx is not None and best_iou >= IOU_THRESHOLD:
        det_bboxes[best_det_idx]['assigned_label'] = gt['label']
        det_bboxes[best_det_idx]['iou'] = best_iou
        used_detections.add(best_det_idx)

# --- Save assigned detections with labels ---
labeled_targets = []
for det in det_bboxes:
    if 'assigned_label' in det:
        labeled_targets.append({
            "bbox": det['bbox'],
            "label": det['assigned_label'],
            "conf": det.get('conf', 0.0),
            "iou": det.get('iou', 0.0)
        })
os.makedirs(os.path.dirname(OUTPUT_TARGETS_PATH), exist_ok=True)
with open(OUTPUT_TARGETS_PATH, 'w') as f:
    json.dump(labeled_targets, f, indent=2)
print(f"[INFO] Saved assigned targets to {OUTPUT_TARGETS_PATH}")

# --- Display the frame with assigned labels ---
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
ret, frame = cap.read()
if not ret:
    raise IOError(f"Could not read frame {frame_num} from video.")

for det in det_bboxes:
    x1, y1, x2, y2 = map(int, det['bbox'])
    conf = det.get('conf', 0.0)
    assigned_label = det.get('assigned_label', None)
    color = (0, 255, 0) if assigned_label else (0, 0, 255)
    label_text = assigned_label if assigned_label else det['label']
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label_text} {conf:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("YOLO Detections with Assigned Labels", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()