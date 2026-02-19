"""
YOLO-based CV layout detection in memory.

Supports:
• Cached YOLO models (no reload per call)
• Bytes or numpy image input
• Optional crop + JSON export
• Clean production-ready structure
"""

import json
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # type: ignore

# Centralized model loaders
from services.paths_models_and_data import (
    get_text_photo_yolo_model,
    get_skill_table_yolo_model,
)


# =========================================================
# Core detection
# =========================================================
def detect_and_segment_cv(
    image_data: Union[np.ndarray, bytes],
    text_photo_model: Optional[YOLO] = None,
    skill_table_model: Optional[YOLO] = None,
    conf_thres: float = 0.20,
    save_path: Optional[Union[str, Path]] = None,
    return_crops: Optional[List[str]] = None,
    return_json: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect layout elements in a CV image.

    Detection classes:
        • text_block
        • photo
        • skill_table (optional second model)

    Parameters
    ----------
    image_data:
        Image in memory (bytes or numpy array).

    text_photo_model:
        Preloaded YOLO model.
        If None → loads cached global model.

    skill_table_model:
        Optional preloaded YOLO model for skill tables.
        If None → loads cached global model only if needed.

    conf_thres:
        YOLO confidence threshold.

    save_path:
        Optional base path for JSON/crop export.

    return_crops:
        List of classes to crop.
        None → disables cropping.

    return_json:
        Whether to export detections as JSON.

    Returns
    -------
    List of filtered detections sorted top→bottom.
    """

    # -----------------------------------------------------
    # Load cached models if not provided
    # -----------------------------------------------------
    if text_photo_model is None:
        text_photo_model = get_text_photo_yolo_model()

    if skill_table_model is None and return_crops and "skill_table" in return_crops:
        skill_table_model = get_skill_table_yolo_model()

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
    def _collect(results, model: YOLO) -> None:
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

                # Optional crop
                if return_crops and cls_name in return_crops:
                    det["crop"] = Image.fromarray(img_rgb[y1:y2, x1:x2])

                detections.append(det)

    # -----------------------------------------------------
    # Run inference — text/photo
    # -----------------------------------------------------
    results_text = text_photo_model.predict(
        source=img_rgb,
        conf=conf_thres,
        verbose=False,
    )
    _collect(results_text, text_photo_model)

    # -----------------------------------------------------
    # Run inference — skill table (optional)
    # -----------------------------------------------------
    if skill_table_model is not None:
        results_skill = skill_table_model.predict(
            source=img_rgb,
            conf=conf_thres,
            verbose=False,
        )
        _collect(results_skill, skill_table_model)

    # -----------------------------------------------------
    # Sort detections (top→bottom, left→right)
    # -----------------------------------------------------
    detections.sort(key=lambda d: (d["y"], d["x"]))

    # -----------------------------------------------------
    # Remove contained boxes of same class
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
    # Optional export
    # -----------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- JSON ----
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

        # ---- Crops ----
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
