"""
In-memory OCR utilities using Tesseract.

Design goals:
- No external model downloads
- Deterministic OCR for clean digital CVs
- Fully compatible with existing pipeline
"""

# =========================================================
# Standard library
# =========================================================
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Union, Optional

# =========================================================
# Third-party
# =========================================================
import numpy as np
from PIL import Image, ImageFilter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # You have to download tesseract e put here the path

# =========================================================
# Project root
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# Typing
# =========================================================
class OCRResult(TypedDict):
    index: int
    label: str
    confidence: float
    bbox: List[int]
    text: str


# =========================================================
# Constants
# =========================================================
BBOX_BUFFER = 18


# =========================================================
# Image helpers
# =========================================================
def _to_pil(image: Union[Image.Image, np.ndarray, bytes]) -> Image.Image:
    """Convert numpy/bytes → PIL.Image safely."""
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, bytes):
        import cv2
        arr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(arr, 1)
        if img is None:
            raise ValueError("Invalid image bytes")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image.astype(np.uint8))

    raise TypeError("Unsupported image type")


def _preprocess_crop(crop: Image.Image) -> Image.Image:
    """
    Light preprocessing improves Tesseract accuracy.
    """
    gray = crop.convert("L")

    # sharpen muito leve
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

    return sharp


# =========================================================
# Core OCR (Tesseract)
# =========================================================
def extract_text_from_bboxes(
    image_data: Union[Image.Image, np.ndarray, bytes],
    detections: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[OCRResult]:

    image = _to_pil(image_data)
    img_w, img_h = image.size

    results: List[OCRResult] = []

    for idx, det in enumerate(detections):

        label = det.get("label") or det.get("class") or ""

        # Skip photos
        if label == "photo":
            continue

        # Get crop
        if "crop" in det and det["crop"] is not None:
            crop = det["crop"]
        else:
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            x1 = max(0, x1 - BBOX_BUFFER)
            y1 = max(0, y1 - BBOX_BUFFER)
            x2 = min(img_w, x2 + BBOX_BUFFER)
            y2 = min(img_h, y2 + BBOX_BUFFER)
            crop = image.crop((x1, y1, x2, y2))

        crop = _preprocess_crop(crop)

        try:
            text = pytesseract.image_to_string(
                crop,
                lang="eng",
                config="--oem 3 --psm 6"
            ).strip()
        except Exception:
            text = ""

        results.append(
            OCRResult(
                index=idx,
                label=label,
                confidence=float(det.get("confidence", 0.0)),
                bbox=list(det.get("bbox", [])),
                text=text,
            )
        )

        if verbose:
            print(f"[{idx}] {label}: {text[:60]}...")

    return results


# =========================================================
# Public helper
# =========================================================
def process_image_with_bbox(
    image_data: Union[Image.Image, np.ndarray, bytes],
    detections: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:

    ocr_results = extract_text_from_bboxes(image_data, detections, verbose)

    text = "\n\n".join(r["text"] for r in ocr_results)

    result = {"text": text, "results": ocr_results}

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"Text saved to: {path}")

    return result


