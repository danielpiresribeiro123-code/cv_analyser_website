"""
Shared OCR core for CV processing pipelines.

Provides:
- PaddleOCR initialization (safe, singleton‑style)
- PyMuPDF raw text extraction
- Full‑document OCR fallback
- Bounding‑box OCR helper (for YOLO full analysis)

This module is intentionally pipeline‑agnostic so both
FAST and FULL analysis import the same OCR backbone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------
# Paddle safe environment configuration (MUST happen before import)
# ---------------------------------------------------------------------
SAFE_PADDLE_DIR = r"C:\paddle_cache"
os.environ.setdefault("USERPROFILE", SAFE_PADDLE_DIR)
os.environ.setdefault("HOME", SAFE_PADDLE_DIR)
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from paddleocr import PaddleOCR  # noqa: E402  (import after env vars)

# ---------------------------------------------------------------------
# Singleton OCR instance
# ---------------------------------------------------------------------
_ocr_instance: PaddleOCR | None = None


def get_ocr() -> PaddleOCR:
    """Return a shared PaddleOCR instance (lazy‑loaded)."""
    global _ocr_instance

    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            enable_mkldnn=False,  # stability on Windows CPU
        )

    return _ocr_instance


# ---------------------------------------------------------------------
# RAW TEXT EXTRACTION — PyMuPDF (fast path)
# ---------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: str | Path) -> str:
    """Extract selectable text directly from PDF."""

    doc = fitz.open(str(pdf_path))
    texts: List[str] = []

    for page in doc:
        page_text = page.get_text()
        if isinstance(page_text, str):
            texts.append(page_text)

    return "\n".join(texts).strip()


# ---------------------------------------------------------------------
# FULL‑PAGE OCR — Paddle fallback
# ---------------------------------------------------------------------

def _pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    """Convert PyMuPDF pixmap to OpenCV BGR numpy array."""

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.h, pix.w, pix.n)

    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def extract_text_paddle_full(pdf_path: str | Path, dpi: int = 300) -> str:
    """Run PaddleOCR over each PDF page image."""

    ocr = get_ocr()
    doc = fitz.open(str(pdf_path))

    text_parts: List[str] = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = _pixmap_to_bgr(pix)

        result = ocr.ocr(img)
        if not result:
            continue

        for line in result:
            for word in line:
                word_text = word[1][0]
                if isinstance(word_text, str):
                    text_parts.append(word_text)

    return " ".join(text_parts).strip()


# ---------------------------------------------------------------------
# PUBLIC RAW TEXT ROUTER
# ---------------------------------------------------------------------

def get_raw_resume_text(pdf_path: str | Path) -> str:
    """
    Try fast extraction first, fallback to OCR if text is too short.
    """

    text = extract_text_pymupdf(pdf_path)

    if len(text) < 20:
        text = extract_text_paddle_full(pdf_path)

    return text


# ---------------------------------------------------------------------
# BOUNDING‑BOX OCR (for FULL analysis pipeline)
# ---------------------------------------------------------------------

def extract_text_from_bboxes(
    image_path: str | Path,
    bbox_data: List[Dict[str, Any]],
    buffer_px: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run OCR inside bounding boxes.

    Parameters
    ----------
    image_path : path to image
    bbox_data  : list of YOLO bbox dicts
    buffer_px  : pixel padding around bbox
    """

    ocr = get_ocr()

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    results: List[Dict[str, Any]] = []

    for item in bbox_data:
        if item.get("class") == "photo":
            continue

        x1, y1, x2, y2 = item["bbox"]

        x1 = max(0, x1 - buffer_px)
        y1 = max(0, y1 - buffer_px)
        x2 = min(img_w, x2 + buffer_px)
        y2 = min(img_h, y2 + buffer_px)

        crop = img.crop((x1, y1, x2, y2))
        crop_array = np.array(crop)

        ocr_result = ocr.predict(crop_array)

        text_lines: List[str] = []

        if ocr_result:
            for res in ocr_result:
                for line in res.get("rec_texts", []):
                    if isinstance(line, str):
                        text_lines.append(line)

        extracted_text = "\n".join(text_lines)

        results.append(
            {
                "index": item.get("index"),
                "class": item.get("class"),
                "confidence": item.get("confidence"),
                "bbox": item.get("bbox"),
                "text": extracted_text,
            }
        )

    return results


# ---------------------------------------------------------------------
# CONCATENATION HELPER (FULL pipeline convenience)
# ---------------------------------------------------------------------

def concat_bbox_text(results: List[Dict[str, Any]]) -> str:
    """Join OCR text from bbox results in reading order."""
    return "\n\n".join(item["text"] for item in results if item.get("text"))


# ---------------------------------------------------------------------
# MODULE TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Core OCR module loaded successfully.")
