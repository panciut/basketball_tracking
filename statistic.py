import cv2
import json
import numpy as np
import os
import config
from utils import iou, find_exact_matching_frame, extract_frame_num

def main():
    COCO_PATH = config.COCO_PATH
    TRACKING_JSON_PATH = config.TRACKING_JSON_PATH
    VIDEO_PATH = config.VIDEO_PATH
    IOU_THRESHOLD = getattr(config, "IOU_THRESHOLD", 0.5)
    stride = 5
    ANNOT_IMAGE_DIR = getattr(config, "ANNOT_IMAGE_DIR", os.path.dirname(config.ANNOT_PATH))

    # Load COCO and parse images/annotations
    with open(COCO_PATH) as f:
        coco = json.load(f)
    id2file = {img['id']: img['file_name'] for img in coco['images']}
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Get video prefix
    video_basename = os.path.basename(VIDEO_PATH)
    video_prefix = video_basename.split('_')[0] if '_' in video_basename else os.path.splitext(video_basename)[0]

    # Find GT images for this video
    gt_img_filenames = [id2file[img['id']] for img in coco['images'] if video_prefix in id2file[img['id']]]
    gt_img_filenames = sorted(gt_img_filenames, key=lambda x: extract_frame_num(x))
    if not gt_img_filenames:
        print(f"[ERROR] No annotation images found for {video_prefix}")
        return

    # Match GT frames to video frames
    first_annot_img_path = os.path.join(ANNOT_IMAGE_DIR, gt_img_filenames[0])
    actual_video_frame = find_exact_matching_frame(first_annot_img_path, VIDEO_PATH, max_search=5)
    print(f"[INFO] First annotation corresponds to video frame {actual_video_frame}")

    # Map: frame_idx -> [GTs]
    gt_per_video_frame = {}
    label_gt = {}  # per-label GT count
    for annotation_index, fname in enumerate(gt_img_filenames):
        video_frame = actual_video_frame + annotation_index * stride
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
            label_gt[label] = label_gt.get(label, 0) + 1

    # Load tracking results
    with open(TRACKING_JSON_PATH) as f:
        tracks_per_frame = json.load(f)
    tracks_per_frame = {int(k): v for k, v in tracks_per_frame.items()}

    # Stats
    label_fp = {}       # False Positives per label
    label_fn = {}       # False Negatives per label
    label_id_switch = {}
    label_lost = {}
    prev_tracker_id_to_label = {}

    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0
    all_matched_ious = []

    # For advanced stats
    label_tracks = {label: [] for label in label_gt}  # list of 0/1 per GT occurrence
    all_gt_ids = set()  # unique GT IDs for fragmentation

    # --- Frame-by-frame stats for annotated frames ---
    framewise_stats = []
    framewise_stats_noball = []
    noball_labels = {"player", "referee"}

    # --- Loop over annotation frames only ---
    for frame_idx in sorted(gt_per_video_frame.keys()):
        gt_objs = gt_per_video_frame[frame_idx]
        tracks = tracks_per_frame.get(frame_idx, [])

        # Prepare match bookkeeping
        matched_gt_idx = set()
        matched_track_idx = set()
        current_tracker_to_label = {}
        IoUs_this_frame = []

        # Try to match GT to tracker outputs by IoU
        id_switches_this_frame = 0
        for gt_idx, gt in enumerate(gt_objs):
            best_iou = 0
            best_track_idx = None
            for t_idx, t in enumerate(tracks):
                if t_idx in matched_track_idx:
                    continue
                iou_score = iou(gt['bbox'], t['bbox'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track_idx = t_idx
            if best_iou >= IOU_THRESHOLD and best_track_idx is not None:
                matched_gt_idx.add(gt_idx)
                matched_track_idx.add(best_track_idx)
                IoUs_this_frame.append(best_iou)
                all_matched_ious.append(best_iou)
                # Assign label to tracker id
                track_id = tracks[best_track_idx]['id']
                current_tracker_to_label[track_id] = gt['label']
                # ID switch detection
                prev_ids = [k for k, v in prev_tracker_id_to_label.items() if v == gt['label']]
                if prev_ids and track_id not in prev_ids:
                    label_id_switch[gt['label']] = label_id_switch.get(gt['label'], 0) + 1
                    id_switches_this_frame += 1
                label_lost.setdefault(gt['label'], 0)
                label_id_switch.setdefault(gt['label'], 0)
                label_tracks[gt['label']].append(1)
            else:
                label_tracks[gt['label']].append(0)
            all_gt_ids.add(gt.get('gt_id'))

        # Count False Negatives: GT not matched to any detection
        frame_fp = 0
        frame_fn = 0
        for gt_idx, gt in enumerate(gt_objs):
            if gt_idx not in matched_gt_idx:
                label_fn[gt['label']] = label_fn.get(gt['label'], 0) + 1
                total_fn += 1
                frame_fn += 1
        # Count False Positives: Tracker detection not matched to any GT
        for t_idx, t in enumerate(tracks):
            if t_idx not in matched_track_idx:
                label = prev_tracker_id_to_label.get(t['id'], "unassigned")
                label_fp[label] = label_fp.get(label, 0) + 1
                total_fp += 1
                frame_fp += 1

        # Count "lost": previous label not seen anymore in GT
        matched_labels = {gt_objs[gt_idx]['label'] for gt_idx in matched_gt_idx}
        for prev_trk_id, prev_label in prev_tracker_id_to_label.items():
            if prev_label not in matched_labels:
                label_lost[prev_label] = label_lost.get(prev_label, 0) + 1

        # Update mapping for next frame
        prev_tracker_id_to_label = current_tracker_to_label.copy()
        total_gt += len(gt_objs)
        total_det += len(tracks)

        # Frame-level summary (ALL)
        num_matches = len(matched_gt_idx)
        avg_iou_this_frame = np.mean(IoUs_this_frame) if IoUs_this_frame else 0.0
        mota = 1.0 - (frame_fp + frame_fn + id_switches_this_frame) / max(1, len(gt_objs))
        framewise_stats.append({
            "frame_idx": frame_idx,
            "GT": len(gt_objs),
            "Det": len(tracks),
            "Matched": num_matches,
            "FP": frame_fp,
            "FN": frame_fn,
            "IDSW": id_switches_this_frame,
            "MOTA": mota,
            "IoU": avg_iou_this_frame
        })

                # --- Frame-level summary (NO BALL, exclude "Ball"/"ball" in GT and det) ---
        # GT indices NOT "Ball" (case-insensitive)
        gt_noball_indices = [i for i, obj in enumerate(gt_objs) if obj["label"].lower() != "ball"]
        # Det indices NOT "ball" (case-insensitive)
        tracks_noball_indices = [i for i, t in enumerate(tracks) if t["label"].lower() != "ball"]
        matched_gt_nb = set()
        matched_tr_nb = set()
        ious_nb = []
        # Simple matching for NB (repeat the matching logic, but on NB sets)
        for ngt_idx, gt_idx in enumerate(gt_noball_indices):
            gt = gt_objs[gt_idx]
            best_iou = 0
            best_t_nb = None
            for ntr_idx, tr_idx in enumerate(tracks_noball_indices):
                if ntr_idx in matched_tr_nb:
                    continue
                t = tracks[tr_idx]
                iou_score = iou(gt['bbox'], t['bbox'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_t_nb = ntr_idx
            if best_iou >= IOU_THRESHOLD and best_t_nb is not None:
                matched_gt_nb.add(ngt_idx)
                matched_tr_nb.add(best_t_nb)
                ious_nb.append(best_iou)
        fn_nb = len(gt_noball_indices) - len(matched_gt_nb)
        fp_nb = len(tracks_noball_indices) - len(matched_tr_nb)
        num_matches_nb = len(matched_gt_nb)
        mota_nb = 1.0 - (fp_nb + fn_nb) / max(1, len(gt_noball_indices))
        avg_iou_nb = np.mean(ious_nb) if ious_nb else 0.0
        framewise_stats_noball.append({
            "frame_idx": frame_idx,
            "GT": len(gt_noball_indices),
            "Det": len(tracks_noball_indices),
            "Matched": num_matches_nb,
            "FP": fp_nb,
            "FN": fn_nb,
            "MOTA": mota_nb,
            "IoU": avg_iou_nb
        })
    # ==== Print summary ====
    print("\n=== SUMMARY ===")
    print(f"Total GT objects: {total_gt}")
    print(f"Total detections: {total_det}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Frames evaluated: {len(gt_per_video_frame)}")
    print(f"FP/frame: {total_fp / max(1, len(gt_per_video_frame)):.2f}")
    print(f"FN/frame: {total_fn / max(1, len(gt_per_video_frame)):.2f}")

    # --- Compute global metrics ---
    overall_precision = (total_gt - total_fn) / total_det if total_det > 0 else 0.0
    overall_recall = (total_gt - total_fn) / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    total_idsw = sum(label_id_switch.values())
    id_switch_rate = total_idsw / max(1, total_gt)
    mota = 1 - (total_fn + total_fp + total_idsw) / max(1, total_gt)
    avg_global_iou = np.mean(all_matched_ious) if all_matched_ious else 0.0

    print(f"\nGlobal Precision: {100*overall_precision:.2f}%")
    print(f"Global Recall:    {100*overall_recall:.2f}%")
    print(f"Global F1-score:  {100*overall_f1:.2f}%")
    print(f"ID switches:      {total_idsw}")
    print(f"ID switch rate:   {100*id_switch_rate:.2f}%")
    print(f"MOTA (rough):     {100*mota:.2f}%")
    print(f"Global Avg IoU:   {100*avg_global_iou:.2f}%")

        # === Compute NB ("no ball") global metrics ===
    total_gt_nb = sum(row["GT"] for row in framewise_stats_noball)
    total_det_nb = sum(row["Det"] for row in framewise_stats_noball)
    total_fp_nb = sum(row["FP"] for row in framewise_stats_noball)
    total_fn_nb = sum(row["FN"] for row in framewise_stats_noball)
    total_matches_nb = sum(row["Matched"] for row in framewise_stats_noball)
    total_idsw_nb = 0  # (not counted in NB, see per-frame for matches only)
    # Compute average IoU over all NB matches
    all_matched_ious_nb = []
    for row in framewise_stats_noball:
        if row["Matched"] > 0:
            all_matched_ious_nb.extend([row["IoU"]] * row["Matched"])
    avg_global_iou_nb = np.mean(all_matched_ious_nb) if all_matched_ious_nb else 0.0

    overall_precision_nb = total_matches_nb / total_det_nb if total_det_nb > 0 else 0.0
    overall_recall_nb = total_matches_nb / total_gt_nb if total_gt_nb > 0 else 0.0
    overall_f1_nb = 2 * overall_precision_nb * overall_recall_nb / (overall_precision_nb + overall_recall_nb) if (overall_precision_nb + overall_recall_nb) > 0 else 0.0
    mota_nb = 1 - (total_fn_nb + total_fp_nb + total_idsw_nb) / max(1, total_gt_nb)

    print(f"\n------ 'NO BALL' (players + referees) METRICS ------")
    print(f"Global Precision (NB): {100*overall_precision_nb:.2f}%")
    print(f"Global Recall (NB):    {100*overall_recall_nb:.2f}%")
    print(f"Global F1-score (NB):  {100*overall_f1_nb:.2f}%")
    print(f"MOTA (NB):             {100*mota_nb:.2f}%")
    print(f"Global Avg IoU (NB):   {100*avg_global_iou_nb:.2f}%")

    # --- Per-label metrics ---
    print("\nPer-label stats:")
    header = f"{'Label':>12} | {'GT':>4} | {'FP':>4} | {'FN':>4} | {'ID_sw':>5} | {'lost':>5} | {'Precision':>9} | {'Recall':>7} | {'F1':>7}"
    print(header)
    print('-'*len(header))
    all_labels = sorted(set(list(label_gt.keys()) + list(label_fp.keys()) + list(label_fn.keys()) +
                           list(label_id_switch.keys()) + list(label_lost.keys())))
    for label in all_labels:
        gt = label_gt.get(label, 0)
        fp = label_fp.get(label, 0)
        fn = label_fn.get(label, 0)
        ids = label_id_switch.get(label, 0)
        lost = label_lost.get(label, 0)
        tp = gt - fn
        dets = tp + fp
        prec = tp / dets if dets > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{label:>12} | {gt:4d} | {fp:4d} | {fn:4d} | {ids:5d} | {lost:5d} | {100*prec:8.2f}% | {100*rec:6.2f}% | {100*f1:6.2f}%")

    # --- Additional Stats: Per-frame averages ---
    print("\nAdditional Stats:")
    avg_gt_per_frame = total_gt / max(1, len(gt_per_video_frame))
    avg_det_per_frame = total_det / max(1, len(gt_per_video_frame))
    print(f"Avg GT objects/frame:     {avg_gt_per_frame:.2f}")
    print(f"Avg detections/frame:     {avg_det_per_frame:.2f}")

    # -- Per-label track statistics: Mostly Tracked, Mostly Lost, Fragmentation --
    print(f"\n{'Label':>12} | {'MT':>4} | {'ML':>4} | {'Frag':>5}")
    print('-'*34)
    for label, history in label_tracks.items():
        if not history:
            continue
        total = len(history)
        tracked = sum(history)
        percent_tracked = tracked / total if total else 0
        MT = int(percent_tracked >= 0.8)
        ML = int(percent_tracked <= 0.2)
        frag = 0
        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                frag += 1
        print(f"{label:>12} | {MT:4d} | {ML:4d} | {frag:5d}")

    # --- Framewise annotated summary ---
    print("\n=== Annotated Frames: Frame-by-frame summary ===")
    print(f"{'Frame':>6} | {'GT':>3} | {'Det':>3} | {'Mch':>3} | {'FP':>2} | {'FN':>2} | {'IDSW':>4} | {'MOTA':>6} | {'IoU':>5} ||| "
          f"{'GT(NB)':>6} | {'Det(NB)':>7} | {'Mch(NB)':>8} | {'FP(NB)':>6} | {'FN(NB)':>6} | {'MOTA(NB)':>9} | {'IoU(NB)':>8}")
    print('-'*120)
    for row, row_nb in zip(framewise_stats, framewise_stats_noball):
        print(f"{row['frame_idx']:6} | {row['GT']:3d} | {row['Det']:3d} | {row['Matched']:3d} | {row['FP']:2d} | {row['FN']:2d} | {row['IDSW']:4d} | {row['MOTA']*100:6.2f} | {row['IoU']*100:5.2f} ||| "
              f"{row_nb['GT']:6d} | {row_nb['Det']:7d} | {row_nb['Matched']:8d} | {row_nb['FP']:6d} | {row_nb['FN']:6d} | {row_nb['MOTA']*100:9.2f} | {row_nb['IoU']*100:8.2f}")

    print("\n=== METRIC DEFINITIONS ===")
    print("GT: Ground-truth objects; Det: Detections; Mch: Matched; FP: False Positive; FN: False Negative; IDSW: ID Switches;")
    print("IoU: mean intersection-over-union for matched pairs (per frame, per summary);")
    print("MOTA = 1 - (FP+FN+IDSW)/GT   (higher is better, up to 100%)")
    print("NB = 'No Ball' subset (players+referee only)")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()