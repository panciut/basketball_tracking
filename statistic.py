import cv2
import json
import numpy as np
import os
import config
import re
from utils import iou, find_exact_matching_frame ,extract_frame_num

def main():
    COCO_PATH = config.COCO_PATH
    TRACKING_JSON_PATH = config.TRACKING_JSON_PATH
    VIDEO_PATH = config.VIDEO_PATH
    IOU_THRESHOLD = getattr(config, "IOU_THRESHOLD", 0.5)
    PAUSE = False
    stride = 5  # Set this appropriately if not always 5
    # Directory with the annotation images
    ANNOT_IMAGE_DIR = getattr(config, "ANNOT_IMAGE_DIR", os.path.dirname(config.ANNOT_PATH))

    # Load COCO and parse images/annotations
    with open(COCO_PATH) as f:
        coco = json.load(f)
    id2file = {img['id']: img['file_name'] for img in coco['images']}
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Get video prefix to match GT images for this video
    video_basename = os.path.basename(VIDEO_PATH)
    video_prefix = video_basename.split('_')[0] if '_' in video_basename else os.path.splitext(video_basename)[0]

    # Find and sort all GT annotation images for the current video
    gt_img_filenames = [id2file[img['id']] for img in coco['images'] if video_prefix in id2file[img['id']]]
    gt_img_filenames = sorted(gt_img_filenames, key=lambda x: extract_frame_num(x))
    if not gt_img_filenames:
        print(f"[ERROR] No annotation images found for {video_prefix}")
        return

    # --- Find true video offset for the first annotation using the utility ---
    first_annot_img_path = os.path.join(ANNOT_IMAGE_DIR, gt_img_filenames[0])
    print(f"[INFO] Matching first annotation {gt_img_filenames[0]} to video {VIDEO_PATH} ...")
    actual_video_frame = find_exact_matching_frame(first_annot_img_path, VIDEO_PATH, max_search=5)
    print(f"[INFO] First annotation corresponds to video frame {actual_video_frame} (1-based indexing)")

    # --- Build: video_frame -> GTs mapping ---
    gt_per_video_frame = {}
    for annotation_index, fname in enumerate(gt_img_filenames):
        video_frame = actual_video_frame + annotation_index * stride  # This is 1-based
        img_id = [img_id for img_id, name in id2file.items() if name == fname][0]
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
        for ann in anns:
            x, y, w, h = ann['bbox']
            label = cat_map[ann['category_id']]
            gt_per_video_frame.setdefault(video_frame, []).append({
                "bbox": [x, y, x + w, y + h],
                "label": label,
                "gt_id": ann['id']
            })
    annotated_frames = sorted(gt_per_video_frame.keys())
    if not annotated_frames:
        print(f"[ERROR] No GT frames found for video prefix '{video_prefix}'.")
        return

    # --- Load tracking results ---
    with open(TRACKING_JSON_PATH) as f:
        tracks_per_frame = json.load(f)
    tracks_per_frame = {int(k): v for k, v in tracks_per_frame.items()}

    # Stats
    tracker_id_to_label = {}
    label_lost_count = {}
    label_id_switch_count = {}
    prev_tracker_id_to_label = {}

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Optional: Output video
    out = None
    output_video_path = "output/annotated_tracking_synced.mp4"

    for frame_idx in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Could not load frame {frame_idx}.")
            break

        tracks = tracks_per_frame.get(frame_idx, [])

        # If this is an annotation frame, update tracker-label mapping and show GT
        if frame_idx in gt_per_video_frame:
            gt_objs = gt_per_video_frame[frame_idx]
            current_tracker_to_label = {}
            matched_gt = set()
            matched_track_ids = set()
            for gt in gt_objs:
                best_iou = 0
                best_track_id = None
                for t in tracks:
                    if t['id'] in matched_track_ids:
                        continue
                    iou_score = iou(gt['bbox'], t['bbox'])
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_track_id = t['id']
                if best_iou >= IOU_THRESHOLD and best_track_id is not None:
                    current_tracker_to_label[best_track_id] = gt['label']
                    matched_gt.add(gt['label'])
                    matched_track_ids.add(best_track_id)
                    # ID switch detection
                    prev_ids = [k for k, v in prev_tracker_id_to_label.items() if v == gt['label']]
                    if prev_ids and best_track_id not in prev_ids:
                        label_id_switch_count[gt['label']] = label_id_switch_count.get(gt['label'], 0) + 1
                    label_lost_count.setdefault(gt['label'], 0)
                    label_id_switch_count.setdefault(gt['label'], 0)
            for prev_trk_id, prev_label in prev_tracker_id_to_label.items():
                if prev_label not in matched_gt:
                    label_lost_count[prev_label] = label_lost_count.get(prev_label, 0) + 1
            prev_tracker_id_to_label = current_tracker_to_label.copy()
            # Draw GTs (only at annotation frames)
            for gt in gt_objs:
                x1, y1, x2, y2 = map(int, gt['bbox'])
                label = gt['label']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
                cv2.putText(frame, label, (x1, y1-18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Always draw tracker output on every frame (with label from last annotation frame)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t['bbox'])
            trk_id = t['id']
            mapped_label = prev_tracker_id_to_label.get(trk_id, "")
            color = (0,255,0) if mapped_label else (0,0,255)
            label_str = mapped_label if mapped_label else "unassigned"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {trk_id} : {label_str}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        t_sec = (frame_idx-1)/fps
        cv2.putText(frame, f"Frame {frame_idx} ({t_sec:.2f}s)", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)

        if out is None:
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        out.write(frame)

        cv2.imshow("Tracking + GT (synced annotations)", frame)
        key = cv2.waitKey(0 if PAUSE else 1)
        if key == 27:
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    print("\n=== SUMMARY ===")
    for label in sorted(label_lost_count.keys()):
        print(f"Label {label}: ID switches={label_id_switch_count[label]}, lost={label_lost_count[label]}")

if __name__ == "__main__":
    main()