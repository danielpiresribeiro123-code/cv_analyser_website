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

from backend.pipeline.vision.yolo_coord_mem import detect_and_segment_cv, MODEL_PATH as YOLO_MODEL_PATH #type: ignore
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# Support functions
# =========================================================

def _create_unique_person_folder(base_path: Path, person_name: str) -> Path:
    """
    Create a unique folder for a candidate inside base_path.

    Example:
        Daniel_Ribeiro/
        Daniel_Ribeiro_1/
        Daniel_Ribeiro_2/
    """

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "", person_name.replace(" ", "_"))
    if not safe_name:
        safe_name = "unknown_person"

    folder = base_path / safe_name

    counter = 1
    while folder.exists():
        folder = base_path / f"{safe_name}_{counter}"
        counter += 1

    folder.mkdir(parents=True, exist_ok=True)
    return folder



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
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline: PDF → Images → CV Check → OCR → Entity Extraction → DataFrame.

    Args:
        pdf_input: Path to a PDF file, or bytes of PDF content.
        save_path: Base directory where a unique folder per candidate will be created.
        cv_confidence_threshold: YOLO confidence threshold for CV detection.
        cv_min_detections: Minimum detections to consider a page as a CV.
        yolo_confidence: YOLO confidence threshold for bbox detection.
        gliner_threshold: GLiNER entity detection threshold.
        verbose: Print progress messages.

    Returns:
        DataFrame combining GLiNER entities and regex-extracted contact info.

    Example:
        >>> df = process_single_cv("resume.pdf", save_path="output/test")
        >>> print(df[['person_name', 'job_title', 'skills', 'email']])
    """

    if verbose:
        print("USING NEW PIPELINE FILE")
        print("=" * 60)
        print("CV PROCESSING PIPELINE")
        print("=" * 60)

    # Step 1: Load PDF
    if verbose:
        print("\n[1/7] Loading PDF...")

    if isinstance(pdf_input, (str, Path)):
        pdf_input = Path(pdf_input)
        if not pdf_input.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_input}")
        with open(pdf_input, "rb") as f:
            pdf_bytes = f.read()
    elif isinstance(pdf_input, bytes):
        pdf_bytes = pdf_input
    else:
        raise TypeError("pdf_input must be a file path or bytes")

    # Step 2: Convert PDF to images
    if verbose:
        print("[2/7] Converting PDF to images...")

    images = convert_pdf_to_images(pdf_bytes)
    if not images:
        raise ValueError("No images extracted from PDF")

    if verbose:
        print(f"     Extracted {len(images)} page(s)")

    # Step 3: Initialize CV filter
    if verbose:
        print("[3/7] Initializing CV filter...")

    cv_filter = CVFilter()

    # Step 4: Process each image
    all_results = []

    for page_num, image in enumerate(images):
        if verbose:
            print(f"\n[4/7] Processing page {page_num + 1}/{len(images)}...")

        # Convert PIL Image to bytes for processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        # Check if image is a CV (run model on in-memory image to avoid file I/O)
        # Convert PIL Image (RGB) to BGR numpy array for model
        img_bgr = np.array(image.convert("RGB"))[:, :, ::-1]

        # Run the already-loaded YOLO model inside CVFilter
        res = cv_filter.model(img_bgr, conf=cv_confidence_threshold)

        is_cv = False
        if res and len(res) > 0 and getattr(res[0], "boxes", None) is not None:
            num_boxes = len(res[0].boxes)
            is_cv = num_boxes >= cv_min_detections

        if not is_cv:
            if verbose:
                print(f"     ❌ Page {page_num + 1} is not a CV, skipping...")
            continue

        if verbose:
            print(f"     ✓ Page {page_num + 1} is a CV")

        # Step 5: Get bounding boxes
        if verbose:
            print("     [5/7] Detecting text regions...")

        detections = detect_and_segment_cv(
            image_data=image_bytes,
            model_path=YOLO_MODEL_PATH,
            conf_thres=yolo_confidence,
            return_crops=True,
            save_json_path=None,)


        if not detections:
            if verbose:
                print(f"     No text regions detected on page {page_num + 1}")
            continue

        if verbose:
            print(f"     Found {len(detections)} text region(s)")

        # Step 6: Extract text via OCR
        if verbose:
            print("     [6/7] Extracting text via OCR...")

        ocr_result = process_image_with_bbox(
            image_data=image_bytes,
            detections=detections,
            output_path=None,
            verbose=False
        )

        ocr_text = ocr_result.get("text", "")


        if not ocr_text.strip():
            if verbose:
                print(f"     No text extracted on page {page_num + 1}")
            continue

        if verbose:
            print(f"     ✓ Extracted {len(ocr_text)} characters")

        # Step 7a: GLiNER Entity Extraction
        if verbose:
            print("     [7a/7] Extracting entities with GLiNER...")

        preprocessed_text = preprocess_ocr_for_gliner(ocr_text)

        gliner_df = process_texts_gliner(
            [preprocessed_text],
            threshold=gliner_threshold
        )

        # Step 7b: Regex-based Contact Info Extraction
        if verbose:
            print("     [7b/7] Extracting contact info via regex...")

        contact_info = extract_contact_info(ocr_text)

        # Combine results
        # Start with GLiNER DataFrame
        if gliner_df.empty:
            result_row = {}
        else:
            result_row = gliner_df.iloc[0].to_dict()

        # Add contact info (regex-extracted)
        emails = contact_info.get("all_emails") or []
        phones = contact_info.get("all_phones") or []
        result_row.update({
            "email": contact_info.get("email"),
            "all_emails": ", ".join(emails),
            "phone": contact_info.get("phone"),
            "all_phones": ", ".join(phones),
            "linkedin": contact_info.get("linkedin"),
            "github": contact_info.get("github"),
            "address": contact_info.get("address"),
            "experience_years": contact_info.get("experience_years"),
        })


        # Add metadata
        result_row["page_num"] = page_num + 1
        result_row["ocr_text_length"] = len(ocr_text)

        all_results.append(result_row)

        if verbose:
            print(f"     ✓ Page {page_num + 1} processed successfully")

    # Combine all results into DataFrame
    if not all_results:
        if verbose:
            print("\n⚠️  No valid CV pages found in PDF")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    person_folder = None

    if save_path is not None:
        base_path = Path(save_path).resolve()
        base_path.mkdir(parents=True, exist_ok=True)

        # obter nome do candidato (fallback caso vazio)
        person_name = (
            df.iloc[0].get("person_name")
            or df.iloc[0].get("person_name_regex")
            or "unknown_person")

        person_folder = _create_unique_person_folder(base_path, person_name)


    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ PIPELINE COMPLETE: {len(df)} CV(s) processed")
        print("=" * 60)

    # Save to CSV
    if person_folder is not None:
        csv_path = person_folder / "results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

        if verbose:
            print(f"\n✓ CSV saved to: {csv_path}")


    return df


def get_combined_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper: Merge GLiNER and regex columns intelligently.

    For columns that exist in both sources (e.g., name/email in future extensions),
    prefer GLiNER output but fall back to regex if GLiNER is missing.

    Args:
        df: DataFrame returned by process_single_cv().

    Returns:
        DataFrame with merged/derived "final" columns.
    """
    if df.empty:
        return df

    merged = df.copy()

    # Define column mappings: (gliner_col, regex_col, merged_col)
    mappings = [
        ("person_name", "person_name_regex", "person_name_final"),
        ("job_title", None, "job_title_final"),
        ("skills", None, "skills_final"),
        ("company", None, "company_final"),
        ("university", None, "university_final"),
        ("degree", None, "degree_final"),
    ]

    for gliner_col, regex_col, merged_col in mappings:
        if gliner_col in merged.columns:
            if regex_col and regex_col in merged.columns:
                merged[merged_col] = merged[gliner_col].fillna(merged[regex_col])
            else:
                merged[merged_col] = merged[gliner_col]

    return merged


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

