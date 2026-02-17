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
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "yolo11s-text-photo-150e4b.pt"


# =========================================================
# Core detection
# =========================================================
def detect_and_segment_cv(
    image_data: Union[np.ndarray, bytes],
    model_path: Union[str, Path],
    conf_thres: float = 0.25,
    return_crops: bool = True,
    save_json_path: Optional[Union[str, Path]] = None,
    save_crops: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect text_blocks and photos in a CV and optionally export bbox JSON
    and cropped images.

    Crops are saved in the same folder as the JSON with descriptive names.
    """

    model = YOLO(model_path)

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

    # -----------------------------------------------------
    # YOLO inference
    # -----------------------------------------------------
    results = model.predict(
        source=img_rgb,
        conf=conf_thres,
        device="cpu",
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # Parse detections
    # -----------------------------------------------------
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

            if return_crops:
                det["crop"] = Image.fromarray(img_rgb[y1:y2, x1:x2])

            detections.append(det)

    # -----------------------------------------------------
    # Sort top→bottom, left→right
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
    if save_json_path is not None:
        save_path = Path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- Save JSON ----
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
        if save_crops:
            base_name = save_path.stem
            crops_dir = save_path.parent / f"{base_name}_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            for idx, d in enumerate(filtered):
                if "crop" not in d:
                    continue

                crop_path = crops_dir / f"{base_name}_{idx:02d}_{d['class']}.png"
                d["crop"].save(crop_path)

    return filtered



# =========================================================
# CLI test
# =========================================================
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python detect_and_segment_cv.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    out_json = PROJECT_ROOT / "outputs" / f"{image_path.stem}_bbox.json"

    detections = detect_and_segment_cv(
        image_data=image_bytes,
        model_path=MODEL_PATH,
        return_crops=False,
        save_json_path=out_json,
    )

    for i, d in enumerate(detections):
        print(i, d["class"], d["bbox"], f"{d['confidence']:.2f}")

    print(f"\nSaved JSON → {out_json}")
