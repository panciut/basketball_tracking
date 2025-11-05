import os
import re
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

INPUT_JSON = "../data/annotations/_annotations.coco.json"
OUTPUT_JSON = "../data/rectified/_annotations_rectified.coco.json"

ALPHA = 0.0

def load_calibration(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist


def extract_image_name(img_entry: Dict[str, Any]) -> str:
    if "file_name" in img_entry and img_entry["file_name"]:
        return img_entry["file_name"]
    extra = img_entry.get("extra", {})
    return extra.get("name", "")


def get_image_size(img_entry: Dict[str, Any]) -> Tuple[int, int]:
    w = img_entry.get("width", None)
    h = img_entry.get("height", None)
    if w is None or h is None:
        raise ValueError(f"Image entry id={img_entry.get('id')} missing width/height; add them to the JSON.")
    return int(w), int(h)


class Rectifier:
    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        # Cache: key=(cam_idx, width, height) -> (mtx, dist, new_mtx)
        self._cache_cam_params: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_cam_params(self, cam_idx: str, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (cam_idx, width, height)
        if key in self._cache_cam_params:
            return self._cache_cam_params[key]

        calib_path = f"../camera_data/cam_{cam_idx}/calib/camera_calib_rectified.json"
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Missing calibration for cam {cam_idx}: {calib_path}")
        mtx, dist = load_calibration(calib_path)

        calib_rect_path = f"../camera_data/cam_{cam_idx}/calib/camera_calib_rectified.json"
        if not os.path.isfile(calib_rect_path):
            raise FileNotFoundError(f"Missing calibration for cam {cam_idx}: {calib_rect_path}")
        rect_mtx, _ = load_calibration(calib_rect_path)

        self._cache_cam_params[key] = (mtx, dist, rect_mtx)
        return mtx, dist, rect_mtx

    def undistort_points(self, pts: np.ndarray, cam_idx: str, width: int, height: int) -> np.ndarray:
        if pts.size == 0:
            return pts.astype(np.float32)

        mtx, dist, rect_mtx = self._get_cam_params(cam_idx, width, height)
        pts_in = pts.reshape(-1, 1, 2).astype(np.float32)
        pts_und = cv2.undistortPoints(pts_in, mtx, dist, R=None, P=rect_mtx)
        return pts_und.reshape(-1, 2)

    @staticmethod
    def clamp_bbox(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, width)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, height)
        return xyxy

    def transform_bbox(self, bbox: List[float], cam_idx: str, width: int, height: int) -> List[float]:
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ], dtype=np.float32)
        und = self.undistort_points(corners, cam_idx, width, height)
        und = self.clamp_bbox(und, width, height)
        xmin = float(np.min(und[:, 0])); ymin = float(np.min(und[:, 1]))
        xmax = float(np.max(und[:, 0])); ymax = float(np.max(und[:, 1]))
        nx = xmin; ny = ymin
        nw = max(0.0, xmax - xmin); nh = max(0.0, ymax - ymin)
        bbox = [nx, ny, nw, nh]
        # Limit bbox coordinates to two decimal places to keep JSON output compact.
        return [float(round(val, 2)) for val in bbox]

    def transform_segmentation(self, seg: List[List[float]], cam_idx: str, width: int, height: int) -> List[List[float]]:
        new_seg: List[List[float]] = []
        for poly in seg:
            if not poly:
                new_seg.append(poly)
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            und = self.undistort_points(pts, cam_idx, width, height)
            und[:, 0] = np.clip(und[:, 0], 0, width)
            und[:, 1] = np.clip(und[:, 1], 0, height)
            new_seg.append(und.reshape(-1).astype(float).tolist())
        return new_seg

    def transform_keypoints(self, kps: List[float], cam_idx: str, width: int, height: int) -> List[float]:
        if not kps:
            return kps
        arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
        pts = arr[:, :2]; vis = arr[:, 2:3]
        mask = vis.flatten() > 0
        if np.any(mask):
            und = self.undistort_points(pts[mask], cam_idx, width, height)
            pts[mask] = und
        pts[:, 0] = np.clip(pts[:, 0], 0, width)
        pts[:, 1] = np.clip(pts[:, 1], 0, height)
        out = np.concatenate([pts, vis], axis=1).reshape(-1).astype(float).tolist()
        return out


def rectify_coco(input_json: str, output_json: str, alpha: float = 0.0) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rectifier = Rectifier(alpha=alpha)

    images: List[Dict[str, Any]] = data.get("images", [])
    annotations: List[Dict[str, Any]] = data.get("annotations", [])

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in images:
        iid = img.get("id", None)
        if iid is None:
            raise ValueError("Each image entry must have an 'id'.")
        img_by_id[int(iid)] = img

    annos_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        img_id = ann.get("image_id", None)
        if img_id is None:
            raise ValueError("Each annotation must have an 'image_id'.")
        annos_by_img[int(img_id)].append(ann)

    updated_annotations: List[Dict[str, Any]] = []

    for idx, (img_id, img) in enumerate(img_by_id.items(), 1):
        name = extract_image_name(img)

        cam_idx = m.group(1) if (m := re.search(r'out[_]?(\d+)', name)) else ""
        width, height = get_image_size(img)

        if not cam_idx:
            print(f"[WARN] Could not infer camera index from image name '{name}' (image_id={img_id}). Skipping rectification for this image.")
            updated_annotations.extend(annos_by_img.get(img_id, []))
            continue

        try:
            _ = rectifier._get_cam_params(cam_idx, width, height)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Skipping image_id={img_id}.")
            updated_annotations.extend(annos_by_img.get(img_id, []))
            continue

        for ann in annos_by_img.get(img_id, []):
            ann_new = dict(ann)

            if "bbox" in ann_new and ann_new["bbox"]:
                ann_new["bbox"] = rectifier.transform_bbox(ann_new["bbox"], cam_idx, width, height)
                x, y, w, h = ann_new["bbox"]
                ann_new["area"] = float(w * h)

            if "segmentation" in ann_new and ann_new["segmentation"]:
                seg = ann_new["segmentation"]
                if isinstance(seg, list) and (len(seg) == 0 or isinstance(seg[0], list)):
                    ann_new["segmentation"] = rectifier.transform_segmentation(seg, cam_idx, width, height)

            if "keypoints" in ann_new and ann_new["keypoints"]:
                ann_new["keypoints"] = rectifier.transform_keypoints(ann_new["keypoints"], cam_idx, width, height)

            updated_annotations.append(ann_new)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(img_by_id)} images...")

    info = data.get("info", {})
    info_rect = dict(info)
    info_rect["description"] = f"{info.get('description', '')} (rectified alpha={alpha})".strip()

    data_out = {
        "info": info_rect,
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": images,
        "annotations": updated_annotations,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False)

    print(f"Saved rectified COCO to: {output_json}")


def main():
    # Usa le variabili di configurazione definite in testa al file.
    input_json = os.path.abspath(INPUT_JSON) if INPUT_JSON else ""
    if not input_json:
        raise ValueError("Imposta INPUT_JSON in testa al file.")

    if OUTPUT_JSON:
        output_json = os.path.abspath(OUTPUT_JSON)
    else:
        base, ext = os.path.splitext(input_json)
        output_json = base + "_rectified" + ext

    rectify_coco(input_json, output_json, alpha=ALPHA)


if __name__ == "__main__":
    main()
