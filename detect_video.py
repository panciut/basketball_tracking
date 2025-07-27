import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO
import config
import warnings

# Optional: suppress warnings (including UserWarnings from some libraries)
warnings.filterwarnings("ignore")

def main():
    MODEL_PATH = config.MODEL_PATH
    VIDEO_PATH = config.VIDEO_PATH
    OUTPUT_PATH = config.OUTPUT_PATH
    DETECTIONS_PATH = config.DETECTIONS_PATH 
    IMG_SIZE = config.IMG_SIZE
    ALLOWED_CLASSES = config.ALLOWED_CLASSES

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    detections_per_frame = {}
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Detecting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Set verbose=False to suppress YOLO per-frame output
        results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]
        frame_detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label not in ALLOWED_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            frame_detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        detections_per_frame[frame_idx] = frame_detections
        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    with open(DETECTIONS_PATH, "w") as f:
        json.dump(detections_per_frame, f, indent=2)
    print(f"Detection complete. Output video saved as {OUTPUT_PATH}")
    print(f"Per-frame detection data saved as {DETECTIONS_PATH}")

if __name__ == "__main__":
    main()