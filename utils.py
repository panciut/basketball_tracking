import cv2
import numpy as np

def find_exact_matching_frame(annot_path, video_path, max_search=5, verbose=True, resize=True):
    """
    Returns the 1-based frame number in video that exactly matches the annotated image,
    or the closest frame among the first `max_search` frames if no exact match.
    """
    ann_img = cv2.imread(annot_path)
    if ann_img is None:
        raise FileNotFoundError(f"Annotation image not found: {annot_path}")
    ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
    h_ann, w_ann = ann_img.shape[:2]

    cap = cv2.VideoCapture(video_path)
    best_frame_idx = None
    best_diff = float('inf')
    exact_match = None

    if verbose:
        print(f"[INFO] Searching for exact match in first {max_search} frames of {video_path} for {annot_path}...")

    for i in range(1, max_search + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize and frame_rgb.shape[:2] != (h_ann, w_ann):
            frame_rgb = cv2.resize(frame_rgb, (w_ann, h_ann))

        diff = np.sum(np.abs(frame_rgb.astype(np.int32) - ann_img.astype(np.int32)))
        if diff < best_diff:
            best_diff = diff
            best_frame_idx = i

        if diff == 0:
            exact_match = i
            if verbose:
                print(f"[RESULT] Exact pixel-wise match found at video frame {i}")
            break

        if verbose:
            print(f"[INFO] Checked frame {i}... Current best: frame {best_frame_idx} (difference {best_diff})")

    cap.release()
    if exact_match is not None:
        return exact_match
    else:
        if verbose:
            print(f"[FINAL] Closest frame in first {max_search}: {best_frame_idx} (pixel sum difference: {best_diff})")
        return best_frame_idx