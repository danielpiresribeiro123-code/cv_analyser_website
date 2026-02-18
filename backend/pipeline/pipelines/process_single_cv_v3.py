"""
End-to-end CV Processing Pipeline.

Takes a PDF, converts it to images, identifies CV pages, extracts text,
and performs entity extraction via both GLiNER and regex patterns.

Returns combined results as a pandas DataFrame and optionally
saves outputs inside a unique candidate folder.
"""

# =========================================================
# Standard library imports
# =========================================================
from pathlib import Path
from typing import Union, Optional
import io
import sys

# =========================================================
# Third-party imports
# =========================================================
import pandas as pd
import numpy as np
import re

# =========================================================
# Local imports (project modules)
# NOTE: These imports assume your package layout is stable.
# If you run this file directly, prefer "python -m ..." to keep imports consistent.
# =========================================================
from backend.pipeline.input.pdf_to_image import convert_pdf_to_images #type: ignore
from backend.pipeline.vision.cv_filter import CVFilter #type: ignore

from backend.pipeline.vision.yolo_coord_mem import detect_and_segment_cv, TEXT_PHOTO_MODEL_PATH, SKILL_TABLE_MODEL_PATH #type: ignore
from backend.pipeline.OCR.ocr_image_mem import process_image_with_bbox #type: ignore

from backend.pipeline.NLP.text_GliNER_preprocessing import preprocess_ocr_for_gliner #type: ignore
from backend.pipeline.NLP.entities_from_text_mem import process_texts_gliner #type: ignore

from backend.pipeline.NLP.contact_info_extraction_mem import extract_contact_info #type: ignore


# =========================================================
# Project root resolution (no path_loader required)
#
# Current file path:
# backend\pipeline\pipelines\process_single_cv.py
#
# parents[0] -> pipelines
# parents[1] -> pipeline
# parents[2] -> backend
# =========================================================
# PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# Support functions
# =========================================================

def _create_unique_cv_folder(
    base_path: Path,
    file_stem: str,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
) -> Path:
    """
    Create a unique folder for a CV using the PDF filename (without extension).

    Example:
        Daniel_Ribeiro_CV_2025/
        Daniel_Ribeiro_CV_2025_1/
    """

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "", file_stem.replace(" ", "_"))
    if not safe_name:
        safe_name = "unknown_cv"

    folder = base_path / safe_name

    counter = 1
    while folder.exists():
        folder = base_path / f"{safe_name}_{counter}"
        counter += 1

    folder.mkdir(parents=True, exist_ok=True)

    # optional subfolders
    if crop_photo:
        (folder / "photos").mkdir(exist_ok=True)

    if crop_skill_table:
        (folder / "skill_tables").mkdir(exist_ok=True)

    return folder



def _save_detection_crops(detections, cv_folder: Path, page_num: int) -> None:
    for idx, det in enumerate(detections):
        if "crop" not in det:
            continue

        cls = det["class"]

        if cls == "photo":
            out_dir = cv_folder / "photos"
        elif cls == "skill_table":
            out_dir = cv_folder / "skill_tables"
        else:
            continue

        out_dir.mkdir(exist_ok=True)
        det["crop"].save(out_dir / f"p{page_num:02d}_{idx:02d}_{cls}.png")


# =========================================================
# Main function
# =========================================================

def process_single_cv(
    pdf_input: Union[str, Path, bytes],
    save_path: Optional[Union[str, Path]] = None,
    cv_confidence_threshold: float = 0.3,
    cv_min_detections: int = 5,
    yolo_confidence: float = 0.25,
    gliner_threshold: float = 0.3,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end CV processing pipeline.

    Pipeline stages:
        1. Load PDF (path or bytes)
        2. Convert PDF pages to images
        3. Identify CV pages using YOLO-based CVFilter
        4. Detect layout regions (text blocks, photos, skill tables)
        5. Extract OCR text from detected regions
        6. Perform entity extraction with GLiNER
        7. Extract contact information via regex
        8. Aggregate results into a structured DataFrame
        9. Optionally save:
            • candidate folder (named after PDF file)
            • cropped photos / skill tables
            • CSV results

    Args:
        pdf_input:
            Path to a PDF file or raw PDF bytes.

        save_path:
            Base directory where a unique folder per CV will be created.
            Folder name is derived from the PDF filename (stem).

        cv_confidence_threshold:
            Confidence threshold used by the CV detection YOLO model.

        cv_min_detections:
            Minimum number of detected boxes required to classify a page as a CV.

        yolo_confidence:
            Confidence threshold for layout detection (text/photo/skill_table).

        gliner_threshold:
            Confidence threshold for GLiNER entity extraction.

        crop_photo:
            If True, detected photo regions are saved.

        crop_skill_table:
            If True, detected skill table regions are saved.

        verbose:
            If True, prints detailed pipeline progress.

    Returns:
        pandas.DataFrame
            One row per processed CV page containing:
                • GLiNER entities
                • regex-extracted contacts
                • metadata (page number, OCR length)

            Returns an empty DataFrame if no valid CV pages are detected.
    """

    # -----------------------------------------------------
    # [1] Load PDF
    # -----------------------------------------------------
    if verbose:
        print("USING NEW PIPELINE FILE")
        print("=" * 60)
        print("CV PROCESSING PIPELINE")
        print("=" * 60)
        print("\n[1/7] Loading PDF...")

    if isinstance(pdf_input, (str, Path)):
        pdf_path = Path(pdf_input)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pdf_bytes = pdf_path.read_bytes()
        pdf_stem = pdf_path.stem
    elif isinstance(pdf_input, bytes):
        pdf_bytes = pdf_input
        pdf_stem = "cv"
    else:
        raise TypeError("pdf_input must be a file path or bytes")

    # -----------------------------------------------------
    # [2] Convert PDF → images
    # -----------------------------------------------------
    if verbose:
        print("[2/7] Converting PDF to images...")

    images = convert_pdf_to_images(pdf_bytes)
    if not images:
        raise ValueError("No images extracted from PDF")

    if verbose:
        print(f"     Extracted {len(images)} page(s)")

    # -----------------------------------------------------
    # [3] Initialize CV detector
    # -----------------------------------------------------
    if verbose:
        print("[3/7] Initializing CV filter...")

    cv_filter = CVFilter()

    # -----------------------------------------------------
    # [4] Prepare output folder (once per CV)
    # -----------------------------------------------------
    cv_folder = None

    if save_path is not None:
        base_path = Path(save_path).resolve()
        base_path.mkdir(parents=True, exist_ok=True)

        cv_folder = _create_unique_cv_folder(
            base_path,
            pdf_stem,
            crop_photo=crop_photo,
            crop_skill_table=crop_skill_table,
        )

    # -----------------------------------------------------
    # [5] Process each page
    # -----------------------------------------------------
    all_results = []

    for page_num, image in enumerate(images):

        if verbose:
            print(f"\n[4/7] Processing page {page_num + 1}/{len(images)}...")

        # Convert PIL image → bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        # Convert to BGR numpy for YOLO CV filter
        img_bgr = np.array(image.convert("RGB"))[:, :, ::-1]

        res = cv_filter.model(img_bgr, conf=cv_confidence_threshold)

        is_cv = False
        if res and len(res) > 0:
            boxes = getattr(res[0], "boxes", None)
            if boxes is not None and len(boxes) >= cv_min_detections:
                is_cv = True


        if not is_cv:
            if verbose:
                print(f"     ❌ Page {page_num + 1} is not a CV, skipping...")
            continue

        if verbose:
            print(f"     ✓ Page {page_num + 1} is a CV")

        # -------------------------------------------------
        # [5] Layout detection + optional crops
        # -------------------------------------------------
        if verbose:
            print("     [5/7] Detecting text regions...")

        crop_classes = []
        if crop_photo:
            crop_classes.append("photo")
        if crop_skill_table:
            crop_classes.append("skill_table")

        detections = detect_and_segment_cv(
            image_data=image_bytes,
            text_photo_model_path=TEXT_PHOTO_MODEL_PATH,
            skill_table_model_path=SKILL_TABLE_MODEL_PATH if crop_skill_table else None,
            conf_thres=yolo_confidence,
            save_path=None,
            return_crops=crop_classes,
            return_json=False,
        )

        if not detections:
            if verbose:
                print(f"     No text regions detected on page {page_num + 1}")
            continue

        if verbose:
            print(f"     Found {len(detections)} text region(s)")

        # Save crops for this page
        if cv_folder is not None:
            _save_detection_crops(detections, cv_folder, page_num + 1)

        # -------------------------------------------------
        # [6] OCR extraction
        # -------------------------------------------------
        if verbose:
            print("     [6/7] Extracting text via OCR...")

        ocr_result = process_image_with_bbox(
            image_data=image_bytes,
            detections=detections,
            output_path=None,
            verbose=False,
        )

        ocr_text = ocr_result.get("text", "")

        if not ocr_text.strip():
            if verbose:
                print(f"     No text extracted on page {page_num + 1}")
            continue

        if verbose:
            print(f"     ✓ Extracted {len(ocr_text)} characters")

        # -------------------------------------------------
        # [7a] GLiNER entity extraction
        # -------------------------------------------------
        if verbose:
            print("     [7a/7] Extracting entities with GLiNER...")

        preprocessed_text = preprocess_ocr_for_gliner(ocr_text)

        gliner_df = process_texts_gliner(
            [preprocessed_text],
            threshold=gliner_threshold,
        )

        # -------------------------------------------------
        # [7b] Regex contact extraction
        # -------------------------------------------------
        if verbose:
            print("     [7b/7] Extracting contact info via regex...")

        contact_info = extract_contact_info(ocr_text)

        # Combine GLiNER + regex
        result_row = {} if gliner_df.empty else gliner_df.iloc[0].to_dict()

        emails = contact_info.get("all_emails") or []
        phones = contact_info.get("all_phones") or []

        result_row.update(
            {
                "email": contact_info.get("email"),
                "all_emails": ", ".join(emails),
                "phone": contact_info.get("phone"),
                "all_phones": ", ".join(phones),
                "linkedin": contact_info.get("linkedin"),
                "github": contact_info.get("github"),
                "address": contact_info.get("address"),
                "experience_years": contact_info.get("experience_years"),
                "page_num": page_num + 1,
                "ocr_text_length": len(ocr_text),
            }
        )

        all_results.append(result_row)

        if verbose:
            print(f"     ✓ Page {page_num + 1} processed successfully")

    # -----------------------------------------------------
    # [8] Build final DataFrame
    # -----------------------------------------------------
    if not all_results:
        if verbose:
            print("\n⚠️  No valid CV pages found in PDF")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # -----------------------------------------------------
    # [9] Save CSV
    # -----------------------------------------------------
    if cv_folder is not None:
        csv_path = cv_folder / "results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

        if verbose:
            print(f"\n✓ CSV saved to: {csv_path}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ PIPELINE COMPLETE: {len(df)} CV page(s) processed")
        print("=" * 60)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        save_path = sys.argv[2] if len(sys.argv) > 2 else None

        df = process_single_cv(pdf_path, save_path=save_path, verbose=True)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df.to_string())
    else:
        print("Usage: python process_single_cv.py <pdf_path> [save_folder]")

