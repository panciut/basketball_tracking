import os
import cv2
import json
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
from utils import iou

def main():
    VIDEO = config.VIDEO
    VIDEO_PATH = config.VIDEO_PATH
    DETECTIONS_PATH = config.DETECTIONS_PATH
    OUTPUT_PATH = f"output/tracked_deepsort_progress_{VIDEO}.mp4"
    TRACKING_JSON_PATH = f"output/tracking_results_{VIDEO}.json"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(DETECTIONS_PATH, "r") as f:
        detections_per_frame = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    tracker = DeepSort(
        max_age=25,                       # Number of frames to keep lost tracks
        n_init=3,                         # Frames required to confirm a track
        nms_max_overlap=1.0,              # NMS overlap for merging detections
        embedder="mobilenet",             # Use mobilenet for embeddings
        half=False,                       # Full precision (set True for speed, False for accuracy)
        bgr=True,                         # OpenCV frames are BGR
        embedder_gpu=False                # Use GPU for embedding
    )

    frame_idx = 1
    pbar = tqdm(total=total_frames, desc="Tracking frames")

    tracking_results = {}  # Store tracking results per frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_detections = detections_per_frame.get(str(frame_idx), [])
        detections = [
            ([det['bbox'][0], det['bbox'][1],
              det['bbox'][2] - det['bbox'][0],
              det['bbox'][3] - det['bbox'][1]],
              det['conf'], det['label']
            )
            for det in frame_detections
        ]

        tracks = tracker.update_tracks(detections, frame=frame)
        frame_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = int(track.track_id)
            
            # --- Assign label via IoU matching ---
            best_iou = 0
            best_label = None
            track_box = [x1, y1, x2, y2]
            for det in frame_detections:
                det_box = det['bbox']
                current_iou = iou(track_box, det_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_label = det['label']
            label = best_label if best_iou > 0.3 else "unassigned"
            
            frame_tracks.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "label": label
            })
            # Draw on frame for video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame, f"ID {track_id} : {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        out.write(frame)
        tracking_results[frame_idx] = frame_tracks
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Save tracking results
    with open(TRACKING_JSON_PATH, "w") as f:
        json.dump(tracking_results, f, indent=2)

    print(f"✔ Tracking done. Saved video: {OUTPUT_PATH}")
    print(f"✔ Tracking data saved as: {TRACKING_JSON_PATH}, {frame_idx - 1} frames processed")

if __name__ == "__main__":
    main()