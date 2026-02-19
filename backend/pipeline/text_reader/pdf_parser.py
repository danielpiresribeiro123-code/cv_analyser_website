# =========================================================
# PDF region text extraction (layout-aware, YOLO-guided)
# =========================================================
from typing import List, Dict, Any, Union, TypedDict, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF


# =========================================================
# Typing
# =========================================================
class PDFRegionResult(TypedDict):
    index: int
    label: str
    confidence: float
    bbox: List[int]
    text: str


# =========================================================
# Helpers
# =========================================================
def _ensure_text(value: Any) -> str:
    """Normalize PyMuPDF outputs into a clean string."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return " ".join(_ensure_text(v) for v in value)

    if isinstance(value, dict):
        parts = []
        for block in value.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    parts.append(span.get("text", ""))
        return " ".join(parts)

    return str(value)


def _open_pdf(pdf_input: Union[str, Path, bytes]) -> fitz.Document:
    """Open PDF from path or bytes."""
    if isinstance(pdf_input, (str, Path)):
        return fitz.open(pdf_input)
    if isinstance(pdf_input, bytes):
        return fitz.open(stream=pdf_input, filetype="pdf")
    raise TypeError("pdf_input must be path or bytes")


# =========================================================
# Core extraction
# =========================================================
def extract_pdf_text_from_bboxes(
    pdf_input: Union[str, Path, bytes],
    detections: List[Dict[str, Any]],
    page_index: int = 0,
    img_size: Optional[Tuple[int, int]] = None,
    verbose: bool = False,
) -> List[PDFRegionResult]:
    """
    Extract text from YOLO `text_block` regions inside a PDF page.

    If no text is found inside the regions,
    returns empty results → caller should fallback to OCR.
    """

    doc = _open_pdf(pdf_input)

    if page_index >= len(doc):
        raise ValueError("Invalid page index")

    page = doc[page_index]

    # -----------------------------------------------------
    # Validate scaling info
    # -----------------------------------------------------
    if img_size is None:
        raise ValueError("img_size must be provided for PDF coordinate scaling")

    img_w, img_h = map(float, img_size)

    pdf_w, pdf_h = page.rect.width, page.rect.height

    scale_x = pdf_w / img_w
    scale_y = pdf_h / img_h

    # -----------------------------------------------------
    # Keep only text blocks
    # -----------------------------------------------------
    text_detections = [
        d for d in detections
        if str(d.get("class")) == "text_block"
    ]

    results: List[PDFRegionResult] = []

    # -----------------------------------------------------
    # Extract region text
    # -----------------------------------------------------
    for idx, det in enumerate(text_detections):

        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        rect = fitz.Rect(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        )

        try:
            raw_text = page.get_text("text", clip=rect)
            text = _ensure_text(raw_text).strip()
        except Exception:
            text = ""

        results.append(
            PDFRegionResult(
                index=idx,
                label="text_block",
                confidence=float(det.get("confidence", 0.0)),
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                text=text,
            )
        )

        if verbose:
            print(f"[PDF region {idx}] {text[:60]}")

    return results


# =========================================================
# High-level wrapper
# =========================================================
def process_pdf_with_bboxes(
    pdf_input: Union[str, Path, bytes],
    detections: List[Dict[str, Any]],
    page_index: int = 0,
    img_size: Optional[Tuple[int, int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Layout-aware PDF text extraction with safe fallbacks.

    Strategy:
    1. Read YOLO text_block regions
    2. If empty → read full page text
    3. If still empty → return empty (pipeline should OCR)
    """

    # -----------------------------------------------------
    # Try region-based extraction
    # -----------------------------------------------------
    region_results = extract_pdf_text_from_bboxes(
        pdf_input=pdf_input,
        detections=detections,
        page_index=page_index,
        img_size=img_size,
        verbose=verbose,
    )

    full_text = "\n\n".join(r["text"] for r in region_results if r["text"])

    # -----------------------------------------------------
    # Fallback 1 → full page text
    # -----------------------------------------------------
    if not full_text.strip():
        try:
            doc = _open_pdf(pdf_input)
            page = doc[page_index]
            full_text = _ensure_text(page.get_text("text")).strip()

            if verbose and full_text:
                print("[PDF fallback] Full page text extracted.")

        except Exception:
            full_text = ""

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------
    return {
        "text": full_text,
        "results": region_results,
    }
