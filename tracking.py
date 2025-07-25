import cv2
import json
import os

# --- CONFIG ---
VIDEO_PATH = "data/video/out13.mp4"
TARGETS_PATH = "output/first_frame_targets.json"
OUTPUT_VIDEO = "output/tracked_opencv.mp4"
TRACKER_TYPE = "CSRT"  # options: 'CSRT', 'KCF', 'MIL', etc.

# --- Load targets (annotated bboxes and labels) ---
with open(TARGETS_PATH) as f:
    targets = json.load(f)  # [{'bbox': [x1, y1, x2, y2], 'label': ...}, ...]

# --- Initialize video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(w), int(h)))

# --- Read the first frame and set up trackers ---
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read the first frame of the video!")

# --- Prepare initial bounding boxes ---
bboxes = []
labels = []
for target in targets:
    x1, y1, x2, y2 = map(int, target["bbox"])
    bboxes.append((x1, y1, x2-x1, y2-y1))  # OpenCV uses (x, y, w, h)
    labels.append(target["label"])

# --- Create MultiTracker ---
multi_tracker = cv2.MultiTracker_create()
for bbox in bboxes:
    if TRACKER_TYPE == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif TRACKER_TYPE == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif TRACKER_TYPE == "MIL":
        tracker = cv2.TrackerMIL_create()
    else:
        raise ValueError(f"Unknown tracker type: {TRACKER_TYPE}")
    multi_tracker.add(tracker, frame, bbox)

# --- Process the video ---
frame_idx = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    success, tracked_boxes = multi_tracker.update(frame)
    for i, newbox in enumerate(tracked_boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        color = (0, 255, 0)
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, labels[i], (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)
    cv2.imshow("OpenCV Multi-Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Tracking complete. Output video saved to {OUTPUT_VIDEO}")