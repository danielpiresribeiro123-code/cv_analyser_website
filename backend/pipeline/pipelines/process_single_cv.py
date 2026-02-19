"""
End-to-end CV Processing Pipeline.

This module:
• Loads a CV PDF and converts it to images or an image
• Identifies valid CV pages using YOLO-based layout density
• Detects structural regions (text blocks, photos, skill tables)
• Extracts OCR text using Tesseract
• Performs entity extraction with GLiNER and regex rules
• Aggregates structured results into a pandas DataFrame
• Optionally saves artifacts inside a unique candidate folder
"""

# =========================================================
# Library imports
# =========================================================
from pathlib import Path
from typing import Union, Optional, Any
import io
import sys
import re
import pandas as pd
import numpy as np
from ultralytics import YOLO #type:ignore
from PIL import Image
import cv2

# =========================================================
# Local pipeline modules
# =========================================================
from pipeline.input.pdf_to_image import convert_pdf_to_images  # type: ignore
from pipeline.vision.cv_filter import CVFilter  # type: ignore
from pipeline.vision.yolo_coord_mem import detect_and_segment_cv  # type: ignore
from pipeline.OCR.ocr_image_mem import process_image_with_bbox  # type: ignore
from pipeline.NLP.text_GliNER_preprocessing import preprocess_ocr_for_gliner  # type: ignore
from pipeline.NLP.entities_from_text_mem import process_texts_gliner  # type: ignore
from pipeline.NLP.contact_info_extraction_mem import extract_contact_info  # type: ignore

# =========================================================
# Support functions
# =========================================================

def _create_unique_cv_folder(
    base_path: Path,
    file_stem: str,
    crop_text_block: bool = False,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
) -> Path:
    """
    Create a unique output folder for a processed CV.

    The folder name is derived from the original PDF filename and
    incremented if a folder with the same name already exists.

    Examples
    --------
    Daniel_Ribeiro_CV_2025/
    Daniel_Ribeiro_CV_2025_1/

    Parameters
    ----------
    base_path : Path
        Root directory where CV folders are stored.
    file_stem : str
        Filename without extension.
    crop_text_block : bool
        Whether to create a subfolder for text block crops.
    crop_photo : bool
        Whether to create a subfolder for photo crops.
    crop_skill_table : bool
        Whether to create a subfolder for skill table crops.

    Returns
    -------
    Path
        Path to the created unique CV folder.
    """

    # -----------------------------------------------------
    # Sanitize filename
    # -----------------------------------------------------
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "", file_stem.replace(" ", "_"))
    if not safe_name:
        safe_name = "unknown_cv"

    # -----------------------------------------------------
    # Ensure uniqueness
    # -----------------------------------------------------
    folder = base_path / safe_name
    counter = 1

    while folder.exists():
        folder = base_path / f"{safe_name}_{counter}"
        counter += 1

    folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Optional subfolders
    # -----------------------------------------------------
    if crop_text_block:
        (folder / "text_blocks").mkdir(exist_ok=True)

    if crop_photo:
        (folder / "photos").mkdir(exist_ok=True)

    if crop_skill_table:
        (folder / "skill_tables").mkdir(exist_ok=True)

    return folder


def _save_detection_crops(
    detections: list[dict],
    cv_folder: Path,
    page_num: int,
) -> None:
    """Save cropped detection images into CV subfolders."""

    if not detections:
        return

    # Pre-create folders once
    dirs = {
        "text_block": cv_folder / "text_blocks",
        "photo": cv_folder / "photos",
        "skill_table": cv_folder / "skill_tables",
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    for idx, det in enumerate(detections):

        crop = det.get("crop")
        cls = det.get("class")

        if crop is None or cls not in dirs:
            continue

        # Ensure PIL
        if not hasattr(crop, "save"):
            continue

        filename = f"p{page_num:02d}_{idx:02d}_{cls}.png"
        crop.save(dirs[cls] / filename)





# =========================================================
# Main function
# =========================================================

def process_single_cv(
    cv_input: Union[str, Path, bytes],
    save_path: Optional[Union[str, Path]] = None,
    cv_confidence_threshold: float = 0.3,
    cv_min_detections: int = 5,
    yolo_confidence: float = 0.25,
    gliner_threshold: float = 0.3,
    crop_text_block: bool = False,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
    analyse_individual_pages: bool = False,
    save_ocr_txt: bool = False,
    save_csv: bool = False,
    gliner_model: Any | None = None,
    filter_yolo_model: YOLO | None = None,
    text_photo_yolo_model: YOLO | None = None,
    skill_table_yolo_model: YOLO | None = None,
    verbose: bool = True,) -> pd.DataFrame:
    """
    End-to-end CV processing pipeline (initial stages).

    Covers:
        [1] PDF loading
        [2] PDF → image conversion
        [3] CV page detection
        [4] Output folder preparation
        [5] Layout detection (YOLO regions)

    Remaining stages handled later in the function.
    """

    # -----------------------------------------------------
    # [1] Load input (PDF or image)
    # -----------------------------------------------------
    if verbose:
        print()
        print("=" * 60)
        print("CV PROCESSING PIPELINE")
        print("=" * 60)
        print("\n[1/7] Loading input...")

    images: list[Image.Image]

    # -----------------------------------------------------
    # Case 1 — Path input
    # -----------------------------------------------------
    if isinstance(cv_input, (str, Path)):
        input_path = Path(cv_input)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        pdf_stem = input_path.stem
        suffix = input_path.suffix.lower()

        # ---------- PDF ----------
        if suffix == ".pdf":
            if verbose:
                print("     Detected PDF file")

            pdf_bytes = input_path.read_bytes()
            images = convert_pdf_to_images(pdf_bytes)

        # ---------- Image ----------
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            if verbose:
                print("     Detected image file")

            with Image.open(input_path) as img:
                images = [img.convert("RGB")]

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # -----------------------------------------------------
    # Case 2 — Bytes input
    # -----------------------------------------------------
    elif isinstance(cv_input, bytes):

        pdf_stem = "cv"

        # PDF magic bytes
        if cv_input.startswith(b"%PDF"):
            if verbose:
                print("     Detected PDF bytes")

            images = convert_pdf_to_images(cv_input)

        else:
            if verbose:
                print("     Detected image bytes")

            nparr = np.frombuffer(cv_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Invalid image bytes")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images = [Image.fromarray(img_rgb)]

    else:
        raise TypeError("cv_input must be a file path or raw bytes")

    # -----------------------------------------------------
    # [2] Validate extracted images
    # -----------------------------------------------------
    if not images:
        raise ValueError("No images could be extracted from input")

    if verbose:
        print(f"\n[2/7] Loaded {len(images)} page(s)/image(s)")

    # -----------------------------------------------------
    # [3] Initialize CV detector
    # -----------------------------------------------------
    if verbose:
        print("\n[3/7] Initializing CV detector...")

    cv_filter = CVFilter(model=filter_yolo_model)

    # -----------------------------------------------------
    # [4] Prepare output folder (once per CV)
    # -----------------------------------------------------
    cv_folder: Optional[Path] = None

    if save_path is not None:
        base_path = Path(save_path).resolve()
        base_path.mkdir(parents=True, exist_ok=True)

        cv_folder = _create_unique_cv_folder(
            base_path=base_path,
            file_stem=pdf_stem,
            crop_text_block=crop_text_block,
            crop_photo=crop_photo,
            crop_skill_table=crop_skill_table,
        )

    # -----------------------------------------------------
    # Prepare accumulators
    # -----------------------------------------------------
    all_results: list[dict[str, Any]] = []
    all_ocr_texts: list[str] = []

    total_text_blocks = 0
    total_photos = 0
    total_skill_tables = 0
    valid_pages = 0
    # -----------------------------------------------------
    # Process each page
    # -----------------------------------------------------
    for page_num, image in enumerate(images, start=1):

        if verbose:
            print(f"\n[4/7] Processing page {page_num}/{len(images)}...")

        # PIL → bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        # PIL → numpy BGR
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # -------------------------------------------------
        # CV detection
        # -------------------------------------------------
        is_cv = cv_filter.is_cv(
            image=img_bgr,
            confidence_threshold=cv_confidence_threshold,
            min_detections=cv_min_detections,
        )

        if not is_cv:
            if verbose:
                print(f"     Page {page_num} skipped (not classified as CV)")
            continue

        if verbose:
            print(f"     Page {page_num} classified as CV")

        # -------------------------------------------------
        # [5] Layout detection
        # -------------------------------------------------
        if verbose:
            print("\n[5/7] Detecting layout regions...")

        crop_classes: list[str] = []

        if crop_text_block:
            crop_classes.append("text_block")
        if crop_photo:
            crop_classes.append("photo")
        if crop_skill_table:
            crop_classes.append("skill_table")

        detections = detect_and_segment_cv(
            image_data=image_bytes,
            text_photo_model=text_photo_yolo_model,
            skill_table_model=skill_table_yolo_model,
            conf_thres=yolo_confidence,
            save_path=None,
            return_crops=crop_classes if crop_classes else None,
            return_json=False,
        )

        if not detections:
            if verbose:
                print(f"     No layout regions detected on page {page_num}")
            continue

        if verbose:
            print(f"     Detected {len(detections)} layout region(s)")

        # -------------------------------------------------
        # Count layout elements
        # -------------------------------------------------
        num_text_blocks = sum(d["class"] == "text_block" for d in detections)
        num_photos = sum(d["class"] == "photo" for d in detections)
        num_skill_tables = sum(d["class"] == "skill_table" for d in detections)

        # -------------------------------------------------
        # Save crops
        # -------------------------------------------------
        if cv_folder is not None:
            _save_detection_crops(detections, cv_folder, page_num)

        # -------------------------------------------------
        # [6] OCR extraction
        # -------------------------------------------------
        if verbose:
            print("\n[6/7] Extracting text via OCR...")

        ocr_result = process_image_with_bbox(
            image_data=image_bytes,
            detections=detections,
            output_path=None,
            verbose=False,
        )

        ocr_text = ocr_result.get("text", "")

        if not ocr_text.strip():
            if verbose:
                print(f"     No text extracted on page {page_num}")
            continue

        if verbose:
            print(f"     Extracted {len(ocr_text)} characters")

        # -------------------------------------------------
        # Accumulate
        # -------------------------------------------------
        all_ocr_texts.append(ocr_text)
        total_text_blocks += num_text_blocks
        total_photos += num_photos
        total_skill_tables += num_skill_tables
        valid_pages += 1


        # -------------------------------------------------
        # Optional per-page analysis
        # -------------------------------------------------
        if analyse_individual_pages:

            # ---- GLiNER ----
            preprocessed_text = preprocess_ocr_for_gliner(ocr_text)

            gliner_df = process_texts_gliner(
                texts=[preprocessed_text],
                model=gliner_model,
                threshold=gliner_threshold,
            )

            # ---- Regex contacts ----
            contact_info = extract_contact_info(ocr_text)

            result_row = {} if gliner_df.empty else gliner_df.iloc[0].to_dict()

            emails = contact_info.get("all_emails") or []
            phones = contact_info.get("all_phones") or []

            result_row.update({
                "email": contact_info.get("email"),
                "all_emails": ", ".join(emails),

                "phone": contact_info.get("phone"),
                "all_phones": ", ".join(phones),

                "address": contact_info.get("address"),
                "experience_years": contact_info.get("experience_years"),

                "languages": ", ".join(contact_info.get("languages", [])),
                "languages_details": contact_info.get("languages_details"),

                "linkedin": contact_info.get("linkedin"),
                "github": contact_info.get("github"),
                "all_websites": ", ".join(contact_info.get("all_websites", [])),
            })

            all_results.append(result_row)

        if verbose:
            print(f"     ✓ Page {page_num} processed successfully")

    # =====================================================
    # [7] Aggregation (document-level) + final DataFrame
    # =====================================================
    if verbose:
        print("\n[7/7] Aggregating results...")

    # -----------------------------------------------------
    # If we are NOT analysing per-page, build a single row
    # -----------------------------------------------------
    if not analyse_individual_pages and valid_pages > 0:
        full_text = "\n".join(all_ocr_texts)
        preprocessed_text = preprocess_ocr_for_gliner(full_text)

        gliner_df = process_texts_gliner(
            texts=[preprocessed_text],
            model=gliner_model,
            threshold=gliner_threshold,
        )

        contact_info = extract_contact_info(full_text)

        result_row = {} if gliner_df.empty else gliner_df.iloc[0].to_dict()

        emails = contact_info.get("all_emails") or []
        phones = contact_info.get("all_phones") or []

        result_row.update({
            "email": contact_info.get("email"),
            "all_emails": ", ".join(emails),

            "phone": contact_info.get("phone"),
            "all_phones": ", ".join(phones),

            "address": contact_info.get("address"),
            "experience_years": contact_info.get("experience_years"),

            "languages": ", ".join(contact_info.get("languages", [])),
            "languages_details": contact_info.get("languages_details"),

            "linkedin": contact_info.get("linkedin"),
            "github": contact_info.get("github"),
            "all_websites": ", ".join(contact_info.get("all_websites", [])),

            "page_num": "all",
            "page_num_total": valid_pages,

            "ocr_text_length": len(full_text),

            "num_text_blocks": total_text_blocks,
            "num_photos": total_photos,
            "num_skill_tables": total_skill_tables,

            "cv_output_folder": str(cv_folder) if cv_folder else None,
        })

        all_results = [result_row]

    # -----------------------------------------------------
    # Optional: save RAW OCR text
    # -----------------------------------------------------
    if save_ocr_txt and cv_folder is not None and valid_pages > 0:
        raw_text_path = cv_folder / "ocr_raw.txt"
        full_raw_text = "\n\n".join(all_ocr_texts)
        raw_text_path.write_text(full_raw_text, encoding="utf-8")

        if verbose:
            print(f"     Raw OCR text saved to: {raw_text_path}")

    # -----------------------------------------------------
    # No valid CV pages
    # -----------------------------------------------------
    if not all_results:
        if verbose:
            print("     No valid CV pages found in input.")
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE: 0 page(s) processed")
            print("=" * 60)
        return pd.DataFrame()

    # -----------------------------------------------------
    # Build DataFrame
    # -----------------------------------------------------
    df = pd.DataFrame(all_results)

    # -----------------------------------------------------
    # Optional: save CSV
    # -----------------------------------------------------
    if save_csv and cv_folder is not None:
        csv_path = cv_folder / "results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

        if verbose:
            print(f"     CSV saved to: {csv_path}")

    # -----------------------------------------------------
    # Final print
    # -----------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        unit = "page(s)" if analyse_individual_pages else "CV(s)"
        print(f"PIPELINE COMPLETE: {len(df)} {unit} processed")
        print("=" * 60)

    return df

