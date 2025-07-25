import os
import cv2
import json

# === CONFIGURATION ===
VIDEO_PATH = 'data/video/out13.mp4'         # Path to the input video
MASKS_DIR = 'output/foreground_masks'       # Where to save the extracted masks
LOG_INTERVAL = 30                           # Interval to print processing status
HISTORY = 250                               # Number of frames for background modeling
VAR_THRESHOLD = 12                          # Mahalanobis distance threshold for foreground
DETECT_SHADOWS = True                       # Whether to detect shadows as foreground
MIN_CONTOUR_AREA = 500                      # Minimum contour area to consider valid

# === INITIALIZATION ===
os.makedirs(MASKS_DIR, exist_ok=True)       # Ensure output directory exists
meta = {}                                   # To store metadata per frame
cap = cv2.VideoCapture(VIDEO_PATH)          # Open video file

if not cap.isOpened():
    print(f"[ERROR] Could not open video: {VIDEO_PATH}")
    exit(1)

mog = cv2.createBackgroundSubtractorMOG2(
    history=HISTORY,
    varThreshold=VAR_THRESHOLD,
    detectShadows=DETECT_SHADOWS
)

print("[INFO] Background subtractor (MOG2) initialized.")
frame_count = 0

# === PROCESSING LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video or read error.")
        break

    frame_count += 1

    # Apply background subtraction
    fg_mask = mog.apply(frame)

    # Basic cleaning: remove noise
    fg_clean = cv2.medianBlur(fg_mask, 5)
    _, fg_thresh = cv2.threshold(fg_clean, 200, 255, cv2.THRESH_BINARY)

    # Save mask to disk
    mask_filename = f"{frame_count:04d}.png"
    mask_path = os.path.join(MASKS_DIR, mask_filename)
    cv2.imwrite(mask_path, fg_thresh)

    meta[frame_count] = {
        "frame": frame_count,
        "mask_path": mask_path
    }

    # Optional display for visual feedback
    cv2.imshow("Foreground Mask", fg_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Display interrupted by user.")
        break

    if frame_count % LOG_INTERVAL == 0:
        print(f"[INFO] Processed {frame_count} frames...")

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()

# Save metadata JSON for later reference
meta_path = os.path.join(MASKS_DIR, "masks_meta.json")
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[INFO] Foreground extraction complete. Masks saved to {MASKS_DIR}")