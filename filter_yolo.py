import os
import cv2
import json

DETECTIONS_PATH = 'output/detections.json'
MASKS_DIR = 'output/foreground_masks'
VIDEO_PATH = 'data/video/out13.mp4'
OUTPUT_FILTERED_PATH = 'output/detections_filtered.json'
OUTPUT_FILTERED_VIDEO = 'output/output_filtered.mp4'
FG_THRESHOLD = 0.10   # Minimum percent of bbox area that must be foreground

def is_detection_on_foreground(bbox, mask, threshold=FG_THRESHOLD):
    x1, y1, x2, y2 = map(int, bbox)
    # Clip to mask boundaries
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, mask.shape[1]), min(y2, mask.shape[0])
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return False
    fg_ratio = (region > 0).sum() / region.size
    return fg_ratio > threshold

# --- Load detections ---
with open(DETECTIONS_PATH, 'r') as f:
    detections = json.load(f)

filtered_detections = {}

# --- Setup video I/O ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_FILTERED_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_numbers = sorted([int(k) for k in detections.keys()])
frame_idx = 0
for frame_num in frame_numbers:
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] Video ended at frame {frame_num}")
        break

    frame_detections = detections[str(frame_num)]
    mask_path = os.path.join(MASKS_DIR, f"{frame_num:04d}.png")
    if not os.path.exists(mask_path):
        print(f"[WARN] Mask {mask_path} not found, skipping mask filtering for frame {frame_num}.")
        filtered_detections[str(frame_num)] = frame_detections
        # Optionally draw all detections here
        for det in frame_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            conf = det['conf']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        out.write(frame)
        cv2.imshow("Filtered Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    mask = cv2.imread(mask_path, 0)  # Load as grayscale
    filtered_detections[str(frame_num)] = []
    for det in frame_detections:
        if is_detection_on_foreground(det['bbox'], mask):
            filtered_detections[str(frame_num)].append(det)
            # Draw only filtered detections
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            conf = det['conf']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    print(f"[INFO] Frame {frame_num}: {len(filtered_detections[str(frame_num)])}/{len(frame_detections)} detections kept.")
    out.write(frame)
    cv2.imshow("Filtered Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Save filtered detections ---
with open(OUTPUT_FILTERED_PATH, 'w') as f:
    json.dump(filtered_detections, f, indent=2)

print(f"[INFO] Filtering complete. Filtered detections saved to {OUTPUT_FILTERED_PATH}")
print(f"[INFO] Filtered output video saved to {OUTPUT_FILTERED_VIDEO}")