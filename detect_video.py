from ultralytics import YOLO
import cv2
import os
import json

# --- CONFIGURATION ---
MODEL_PATH = "yolov8l.pt"
VIDEO_PATH = "data/video/out13.mp4"
OUTPUT_PATH = "output/output.mp4"
DETECTIONS_PATH = "output/detections.json"
IMG_SIZE = 1280
ALLOWED_CLASSES = ["person", "sports ball"]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load YOLOv8 large model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

detections_per_frame = {}
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # --- Run YOLO with increased input resolution ---
    results = model(frame, imgsz=IMG_SIZE)[0]

    frame_detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label not in ALLOWED_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Save detection info
        frame_detections.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "conf": conf
        })

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Store detections for this frame
    detections_per_frame[frame_idx] = frame_detections

    out.write(frame)

cap.release()
out.release()

# Save all detections to JSON
with open(DETECTIONS_PATH, "w") as f:
    json.dump(detections_per_frame, f, indent=2)

print(f"Detection complete. Output video saved as {OUTPUT_PATH}")
print(f"Per-frame detection data saved as {DETECTIONS_PATH}")