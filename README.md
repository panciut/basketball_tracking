# Basketball Multi-Object Tracking Pipeline

This project provides a modular pipeline for **detecting, tracking, and evaluating** objects (players, referees, ball) in basketball game videos, using YOLOv8 and DeepSORT.

---

## **Requirements**

**Python 3.9-3.12** (recommended 3.11)

### **Install dependencies**

```bash
pip install -r requirements.txt
```

**Minimal packages needed** (if no `requirements.txt` is present):

* `opencv-python`
* `numpy`
* `tqdm`
* `ultralytics`
* `deep_sort_realtime`
* (optional, if using Jupyter: `notebook`)

---

## **Data Organization**

Your project directory should be organized as follows:

```
basketball_tracking/
│
├── data/
│   ├── video/
│   │   └── out13.mp4              # Video files (from professor drive)
│   └── annotations/
│       ├── out13_frame_0001_....jpg   # Annotated frame images from Roboflow
│       └── _annotations.coco.json     # COCO format annotation file (from Roboflow export)
│
├── output/
│   └── ...                        # All outputs saved here (videos, jsons)
│
├── config.py
├── detect_video.py
├── track_deepsort.py
├── statistic.py
├── utils.py
├── README.md
```

* **Videos:** Should be placed in `data/video/` (e.g., `out13.mp4`).
* **Annotations:** Export your dataset from Roboflow as **COCO format** (include both `.json` and frame images) into `data/annotations/`.

  * You need both the **COCO JSON** file and the corresponding annotated images.

---

## **Pipeline Steps**

### **1. Detection (detect\_video.py)**

**Purpose:** Runs YOLOv8 on each frame to detect objects.

**Input:**

* `data/video/out13.mp4`
* Model weights (`yolov8x.pt` by default; see `config.py`)
* `config.py` for other settings

**Output:**

* Detection video with boxes (`output/outputout13.mp4`)
* Per-frame detection results in JSON (`output/detections.json`)

**Usage:**

```bash
python detect_video.py
```

* Progress bar shown with tqdm.
* Only classes listed in `ALLOWED_CLASSES` in `config.py` are detected.
* Edit `config.py` to change video, model, or class list.

---

### **2. Tracking (track\_deepsort.py)**

**Purpose:** Tracks detected objects using DeepSORT.

**Input:**

* Per-frame detection JSON from step 1 (`output/detections.json`)
* Same video file (`data/video/out13.mp4`)

**Output:**

* Tracked video (`output/tracked_deepsort_progress_out13.mp4`)
* Frame-by-frame tracker results (`output/tracking_results_out13.json`)

**Usage:**

```bash
python track_deepsort.py
```

* Progress bar shown.
* Each detected object is assigned an ID, tracked across frames.

---


### **3. Statistic / Evaluation (statistic.py)**

**Purpose:** Evaluates tracking performance using the annotated frames, shows synchronized tracking and annotation.

**Input:**

* COCO annotation file and annotated images
* Tracking results JSON (`output/tracking_results_out13.json`)
* Video file

**Output:**

* Synchronized video with tracking and ground truth (`output/annotated_tracking_synced.mp4`)
* On-screen display of tracker IDs and assigned labels, and annotation bounding boxes at annotated frames.
* Summary of **ID switches** and **lost tracks** for each label.

**Usage:**

```bash
python statistic.py
```

* Shows a window with live comparison of tracking and annotation, saves annotated video.
* Prints a summary (e.g., `Label Red_11: ID switches=3, lost=2`) at the end.

---

## **Input Data Details**

* **Annotations and frames**:
  Must be exported from Roboflow as **COCO format**:

  * `data/annotations/_annotations.coco.json`
  * Annotated frame images (jpg/png, must match the video being processed).

* **Video files**:
  Must be placed in `data/video/`. Videos are usually provided by the professor (shared drive or local).

---

## **Configuration**

Edit `config.py` to set:

* Which video to use (`VIDEO`, `VIDEO_PATH`)
* Model weights (`MODEL_PATH`)
* Allowed detection classes (`ALLOWED_CLASSES`)
* Paths to annotation files

Change `RUN_...` flags to skip or run different steps as needed.

---

## **Example Run**

1. Set up your `config.py` (select video, model, etc).
2. Run detection:

   ```bash
   python detect_video.py
   ```
3. Run tracking:

   ```bash
   python track_deepsort.py
   ```
4. (Optional) Run mapping for assignment checks:

   ```bash
   python mapping.py
   ```
5. Run evaluation/statistics:

   ```bash
   python statistic.py
   ```

---

## **Troubleshooting**

* **Module not found:** Install missing dependencies with `pip install ...`.
* **Incorrect matches:** Check that your annotation images correspond exactly to the correct video frames.
* **COCO JSON errors:** Ensure Roboflow export includes all image files and the JSON.

---

## **Notes**

* For best results, ensure all paths in `config.py` are correct and up to date.
* The evaluation/statistics script expects **frame-perfect synchronization** between annotation images and video. If frames are off, use the `find_exact_matching_frame` utility (see `utils.py`).
---

**For more info or debugging, check the comments in each script.**
If you need to fine-tune YOLO on your data, ask for an extended guide.
