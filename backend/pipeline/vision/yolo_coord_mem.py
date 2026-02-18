import sys
import json
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # type: ignore


# =========================================================
# Project root resolution
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_PHOTO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "yolo11s-text-photo-150e4b.pt"
SKILL_TABLE_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "yolo11s-skill_table-150e4b.pt"

# =========================================================
# Core detection
# =========================================================
def detect_and_segment_cv(
    image_data: Union[np.ndarray, bytes],
    text_photo_model_path: Union[str, Path],
    skill_table_model_path: Optional[Union[str, Path]] = None,
    conf_thres: float = 0.20,
    save_path: Optional[Union[str, Path]] = None,
    return_crops: Optional[List[str]] = ["photo"],
    return_json: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect text_blocks, photos and optionally skill_tables in a CV image.

    Features
    --------
    - Supports two YOLO models:
        1) text/photo detector (mandatory)
        2) skill table detector (optional)
    - Returns structured detections sorted top→bottom.
    - Optionally exports:
        • JSON with bounding boxes
        • Cropped images per detection class

    Parameters
    ----------
    image_data : np.ndarray | bytes
        Image in memory.
    text_photo_model_path : str | Path
        Path to YOLO model detecting text/photo.
    skill_table_model_path : str | Path | None
        Optional YOLO model detecting skill tables.
    conf_thres : float
        Confidence threshold.
    save_path : str | Path | None
        Base path for JSON/crops.
    return_crops : list[str] | None
        Classes to crop. None disables cropping.
    return_json : bool
        Whether to export JSON.

    Returns
    -------
    List[Dict]
        Filtered detections.
    """

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------
    text_photo_model = YOLO(text_photo_model_path)
    skill_table_model = YOLO(skill_table_model_path) if skill_table_model_path else None

    # -----------------------------------------------------
    # Decode image
    # -----------------------------------------------------
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes")

    elif isinstance(image_data, np.ndarray):
        img_bgr = image_data

    else:
        raise TypeError(f"Unsupported image_data type: {type(image_data)}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    detections: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # Helper to parse YOLO results
    # -----------------------------------------------------
    def _collect_detections(results, model):
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = float(box.conf.item())

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                det: Dict[str, Any] = {
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "y": y1,
                    "x": x1,
                }

                # Add crop only if requested
                if return_crops and cls_name in return_crops:
                    det["crop"] = Image.fromarray(img_rgb[y1:y2, x1:x2])

                detections.append(det)

    # -----------------------------------------------------
    # Run inference — text/photo
    # -----------------------------------------------------
    results_text = text_photo_model.predict(
        source=img_rgb,
        conf=conf_thres,
        device="cpu",
        verbose=False,
    )
    _collect_detections(results_text, text_photo_model)

    # -----------------------------------------------------
    # Run inference — skill table (optional)
    # -----------------------------------------------------
    if skill_table_model is not None:
        results_skill = skill_table_model.predict(
            source=img_rgb,
            conf=conf_thres,
            device="cpu",
            verbose=False,
        )
        _collect_detections(results_skill, skill_table_model)

    # -----------------------------------------------------
    # Sort detections top→bottom, left→right
    # -----------------------------------------------------
    detections.sort(key=lambda d: (d["y"], d["x"]))

    # -----------------------------------------------------
    # Remove contained boxes (same class)
    # -----------------------------------------------------
    filtered: List[Dict[str, Any]] = []

    for i, det_i in enumerate(detections):
        x1_i, y1_i, x2_i, y2_i = det_i["bbox"]
        contained = False

        for j, det_j in enumerate(detections):
            if i == j or det_i["class"] != det_j["class"]:
                continue

            x1_j, y1_j, x2_j, y2_j = det_j["bbox"]

            if (
                x1_i >= x1_j
                and y1_i >= y1_j
                and x2_i <= x2_j
                and y2_i <= y2_j
                and det_i["bbox"] != det_j["bbox"]
            ):
                contained = True
                break

        if not contained:
            filtered.append(det_i)

    # -----------------------------------------------------
    # Optional JSON + crop export
    # -----------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- Save JSON ----
        if return_json:
            serializable = [
                {
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "bbox": list(d["bbox"]),
                }
                for d in filtered
            ]

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)

        # ---- Save crops ----
        if return_crops:
            base_name = save_path.stem
            crops_dir = save_path.parent / f"{base_name}_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            for idx, d in enumerate(filtered):
                if "crop" not in d:
                    continue

                crop_path = crops_dir / f"{base_name}_{idx:02d}_{d['class']}.png"
                d["crop"].save(crop_path)

    return filtered
