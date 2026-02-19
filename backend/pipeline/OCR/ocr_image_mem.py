"""
In-memory OCR utilities using Tesseract.

Design principles
-----------------
• No external model downloads
• Deterministic OCR for clean digital CVs
• Fully compatible with modular CV pipeline
• Centralized Tesseract configuration via services.paths_models_and_data
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
import cv2  # type: ignore

# =========================================================
# Project services
# =========================================================
from services.paths_models_and_data import run_tesseract_ocr


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
    """
    Convert numpy array or raw bytes into a PIL.Image safely.
    """
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, bytes):
        arr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image bytes")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image.astype(np.uint8))

    raise TypeError("Unsupported image type for OCR")


def _preprocess_crop(crop: Image.Image) -> Image.Image:
    """
    Apply lightweight preprocessing to improve OCR accuracy.

    Steps:
    • Convert to grayscale
    • Apply mild sharpening
    """
    gray = crop.convert("L")

    sharp = gray.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2)
    )

    return sharp


# =========================================================
# Core OCR extraction
# =========================================================
def extract_text_from_bboxes(
    image_data: Union[Image.Image, np.ndarray, bytes],
    detections: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[OCRResult]:
    """
    Extract OCR text from detected bounding boxes.

    Parameters
    ----------
    image_data
        Source image in memory.
    detections
        YOLO detections containing bbox, class, confidence, and optional crop.
    verbose
        If True, prints partial OCR output for debugging.

    Returns
    -------
    List[OCRResult]
        Structured OCR results per detection.
    """

    image = _to_pil(image_data)
    img_w, img_h = image.size

    results: List[OCRResult] = []

    for idx, det in enumerate(detections):

        label = det.get("label") or det.get("class") or ""

        # Skip non-textual regions
        if label == "photo":
            continue

        # -------------------------------------------------
        # Determine crop region
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Run Tesseract OCR (centralized config)
        # -------------------------------------------------
        try:
            text = run_tesseract_ocr(
                crop,
                lang="eng",
                config="--oem 3 --psm 6",
            )
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
# Public pipeline helper
# =========================================================
def process_image_with_bbox(
    image_data: Union[Image.Image, np.ndarray, bytes],
    detections: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    High-level OCR processing wrapper.

    Returns:
        {
            "text": full_text,
            "results": structured_ocr_results
        }
    """

    ocr_results = extract_text_from_bboxes(image_data, detections, verbose)

    full_text = "\n\n".join(r["text"] for r in ocr_results)

    result = {
        "text": full_text,
        "results": ocr_results,
    }

    # -----------------------------------------------------
    # Optional JSON export
    # -----------------------------------------------------
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": full_text}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"OCR text saved to: {path}")

    return result
