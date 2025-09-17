import json
from collections import Counter, defaultdict
import config

def summarize_detections(detections_path=None):
    if detections_path is None:
        detections_path = config.DETECTIONS_PATH

    with open(detections_path, "r") as f:
        detections_per_frame = json.load(f)

    total_frames = len(detections_per_frame)
    total_detections = 0
    label_counter = Counter()
    frames_with_zero = 0
    frames_per_label = defaultdict(list)

    print(f"Loaded detections for {total_frames} frames from {detections_path}\n")

    for frame, dets in detections_per_frame.items():
        n = len(dets)
        total_detections += n
        if n == 0:
            frames_with_zero += 1
        labels_in_frame = set()
        for d in dets:
            label = d.get("label", "unknown")
            label_counter[label] += 1
            labels_in_frame.add(label)
        for label in labels_in_frame:
            frames_per_label[label].append(frame)

    print(f"Total frames:           {total_frames}")
    print(f"Total detections:       {total_detections}")
    print(f"Average/frame:          {total_detections/total_frames:.2f}")
    print(f"Frames with 0 detections: {frames_with_zero}\n")

    print("Detections per label:")
    for label, count in label_counter.most_common():
        print(f"  {label:12}: {count:4d}  (in {len(frames_per_label[label])} frames)")

    print("\nSample frames for each label:")
    for label, frames in frames_per_label.items():
        print(f"  {label:12}: {', '.join(frames[:5])}{' ...' if len(frames) > 5 else ''}")

    # Uncomment for per-frame breakdown
    # for frame, dets in detections_per_frame.items():
    #     print(f"Frame {frame}: {[(d['label'], d['conf']) for d in dets]}")

if __name__ == "__main__":
    summarize_detections()