import cv2
import json
import numpy as np

VIDEO_PATH = "data/video/out13.mp4"
DETECTIONS_PATH = "output/detections_filtered.json"  # or detections.json if you prefer
TARGETS_PATH = "output/first_frame_targets.json"
OUTPUT_VIDEO = "output/tracked_labeled_trajectory.mp4"
IOU_THRESHOLD = 0.3
LOST_TOLERANCE = 10    # Frames allowed to be lost before a track is dropped
RECOVER_RADIUS = 70    # Pixel radius to search near predicted position

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

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

with open(DETECTIONS_PATH) as f:
    detections = json.load(f)
with open(TARGETS_PATH) as f:
    targets = json.load(f)

# 1. Find the mapped (annotated) frame number
frame_numbers = sorted([int(k) for k in detections.keys()])
mapped_frame = frame_numbers[0]
while str(mapped_frame) not in detections:
    mapped_frame += 1

# 2. Initialize tracks
tracks = []
for i, obj in enumerate(targets):
    x1, y1, x2, y2 = obj["bbox"]
    cx, cy = get_center(obj["bbox"])
    tracks.append({
        "label": obj["label"],
        "bbox": obj["bbox"],
        "id": i + 1,
        "lost": 0,
        "center_hist": [(cx, cy)]
    })

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(w), int(h)))

# --- Move video to mapped frame
start_idx = frame_numbers.index(mapped_frame)
for _ in range(start_idx):
    cap.read()

for idx in range(start_idx, len(frame_numbers)):
    frame_num = frame_numbers[idx]
    ret, frame = cap.read()
    if not ret:
        break
    frame_dets = detections[str(frame_num)]

    assigned = set()
    # 1. Match tracks to detections using IoU
    for track in tracks:
        best_iou = 0
        best_det = None
        for det_idx, det in enumerate(frame_dets):
            if det_idx in assigned:
                continue
            iou_score = iou(track["bbox"], det["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_det = det_idx
        if best_iou >= IOU_THRESHOLD and best_det is not None:
            track["bbox"] = frame_dets[best_det]["bbox"]
            cx, cy = get_center(track["bbox"])
            track["center_hist"].append((cx, cy))
            if len(track["center_hist"]) > 10:
                track["center_hist"] = track["center_hist"][-10:]
            track["lost"] = 0
            assigned.add(best_det)
        else:
            track["lost"] += 1

    # 2. Try to recover lost tracks by trajectory prediction
    for track in tracks:
        if track["lost"] > 0 and track["lost"] <= LOST_TOLERANCE:
            hist = track["center_hist"]
            if len(hist) >= 2:
                (cx2, cy2), (cx1, cy1) = hist[-1], hist[-2]
                vx, vy = cx2 - cx1, cy2 - cy1
            else:
                vx, vy = 0, 0
            pred_cx = hist[-1][0] + vx
            pred_cy = hist[-1][1] + vy

            min_dist = float("inf")
            best_det = None
            for det_idx, det in enumerate(frame_dets):
                if det_idx in assigned:
                    continue
                det_cx, det_cy = get_center(det["bbox"])
                dist = np.sqrt((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2)
                if dist < min_dist and dist < RECOVER_RADIUS:
                    min_dist = dist
                    best_det = det_idx
            if best_det is not None:
                track["bbox"] = frame_dets[best_det]["bbox"]
                new_cx, new_cy = get_center(track["bbox"])
                track["center_hist"].append((new_cx, new_cy))
                if len(track["center_hist"]) > 10:
                    track["center_hist"] = track["center_hist"][-10:]
                track["lost"] = 0
                assigned.add(best_det)
            else:
                # Still lost, but predict the next center
                track["center_hist"].append((pred_cx, pred_cy))
                if len(track["center_hist"]) > 10:
                    track["center_hist"] = track["center_hist"][-10:]

    # 3. Draw active tracks
    for track in tracks:
        if track["lost"] <= LOST_TOLERANCE:
            x1, y1, x2, y2 = map(int, track["bbox"])
            color = (0, 255, 0) if track["lost"] == 0 else (0, 255, 255)
            label_text = f"{track['label']} | ID:{track['id']}"
            if track["lost"] > 0:
                label_text += f" [LOST {track['lost']}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)
    cv2.imshow("Trajectory-Aware Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Tracking complete. Output video saved to {OUTPUT_VIDEO}")