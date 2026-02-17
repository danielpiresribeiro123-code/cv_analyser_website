"""
End-to-end CV Processing Pipeline.

Takes a PDF, converts to images, identifies CVs, extracts text, 
and performs entity extraction via both GLiNER and regex patterns.

Returns combined results as a pandas DataFrame with option to save as CSV.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import pandas as pd

from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]


def process_single_cv(
    pdf_input: Union[str, Path, bytes],
    save_csv: Optional[Union[str, Path]] = None,
    cv_confidence_threshold: float = 0.3,
    cv_min_detections: int = 5,
    yolo_confidence: float = 0.25,
    gliner_threshold: float = 0.3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline: PDF → Images → CV Check → OCR → Entity Extraction → DataFrame.
    
    Args:
        pdf_input: Path to PDF file, or bytes of PDF content
        save_csv: Optional path to save results as CSV
        cv_confidence_threshold: YOLO confidence threshold for CV detection
        cv_min_detections: Minimum detections to consider image a CV
        yolo_confidence: YOLO confidence threshold for bbox detection
        gliner_threshold: GLiNER entity detection threshold
        verbose: Print progress messages
    
    Returns:
        DataFrame combining GLiNER entities and regex-extracted contact info
    
    Example:
        >>> df = process_single_cv("resume.pdf", save_csv="results.csv")
        >>> print(df[['person_name', 'job_title', 'skills', 'email']])
    """
    
    if verbose:
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
        raise TypeError("pdf_input must be file path or bytes")
    
    # Step 2: Convert PDF to images
    if verbose:
        print("[2/7] Converting PDF to images...")
    # Lazy import convert_pdf_to_images
    from input_processing.pdf_to_image import convert_pdf_to_images

    images = convert_pdf_to_images(pdf_bytes)
    if not images:
        raise ValueError("No images extracted from PDF")
    
    if verbose:
        print(f"     Extracted {len(images)} page(s)")
    
    # Step 3: Initialize CV filter
    if verbose:
        print("[3/7] Initializing CV filter...")
    # Lazy import CVFilter and instantiate
    from input_processing.cv_filter import CVFilter
    cv_filter = CVFilter()
    
    # Step 4: Process each image
    all_results = []
    
    for page_num, image in enumerate(images):
        if verbose:
            print(f"\n[4/7] Processing page {page_num + 1}/{len(images)}...")
        
        # Convert PIL Image to bytes for processing
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Check if image is a CV (run model on in-memory image to avoid file I/O)
        import numpy as _np
        # Convert PIL Image (RGB) to BGR numpy array for model
        img_bgr = _np.array(image.convert("RGB"))[:, :, ::-1]
        # Run the already-loaded YOLO model inside CVFilter
        res = cv_filter.model(img_bgr, conf=cv_confidence_threshold)
        is_cv = False
        if res and len(res) > 0 and getattr(res[0], 'boxes', None) is not None:
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
            print(f"     [5/7] Detecting text regions...")
        # Lazy-import YOLO detection function and model path
        from ocr.yolo_coord_mem import detect_and_segment_cv, MODEL_PATH as YOLO_MODEL_PATH

        detections = detect_and_segment_cv(
            image_data=image_bytes,
            model_path=YOLO_MODEL_PATH,
            conf_thres=yolo_confidence,
            return_crops=True
        )
        
        if not detections:
            if verbose:
                print(f"     No text regions detected on page {page_num + 1}")
            continue
        
        if verbose:
            print(f"     Found {len(detections)} text region(s)")
        
        # Step 6: Extract text via OCR
        if verbose:
            print(f"     [6/7] Extracting text via OCR...")
        # Lazy-import OCR processor
        from ocr.ocr_image_mem import process_image_with_bbox

        ocr_result = process_image_with_bbox(
            image_data=image_bytes,
            detections=detections,
            output_path=None,
            verbose=False
        )
        
        ocr_text = ocr_result["text"]
        
        if not ocr_text.strip():
            if verbose:
                print(f"     No text extracted on page {page_num + 1}")
            continue
        
        if verbose:
            print(f"     ✓ Extracted {len(ocr_text)} characters")
        
        # Step 7a: GLiNER Entity Extraction
        if verbose:
            print(f"     [7a/7] Extracting entities with GLiNER...")
        # Lazy-import GLiNER preprocessing and extractor
        from GliNER.text_GliNER_preprocessing import preprocess_ocr_for_gliner
        from GliNER.entities_from_text_mem import process_texts_gliner

        # Preprocess for GLiNER
        preprocessed_text = preprocess_ocr_for_gliner(ocr_text)

        # Extract entities
        gliner_df = process_texts_gliner(
            [preprocessed_text],
            threshold=gliner_threshold
        )
        
        # Step 7b: Regex-based Contact Info Extraction
        if verbose:
            print(f"     [7b/7] Extracting contact info via regex...")
        # Lazy-import regex extractor (lightweight)
        from info_extraction.contact_info_extraction_mem import extract_contact_info
        contact_info = extract_contact_info(ocr_text)
        
        # Combine results
        # Start with GLiNER DataFrame
        result_row = gliner_df.iloc[0].to_dict()
        
        # Add contact info (regex-extracted)
        result_row.update({
            "email_regex": contact_info["email"],
            "all_emails_regex": ", ".join(contact_info["all_emails"]),
            "phone_regex": contact_info["phone"],
            "all_phones_regex": ", ".join(contact_info["all_phones"]),
            "linkedin_regex": contact_info["linkedin"],
            "github_regex": contact_info["github"],
            "address_regex": contact_info["address"],
            "experience_years_regex": contact_info["experience_years"],
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
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ PIPELINE COMPLETE: {len(df)} CV(s) processed")
        print("=" * 60)
    
    # Optional: Save to CSV (saved under src/outputs, semicolon-delimited)
    if save_csv is not None:
        outputs_dir = Path(PROJECT_ROOT) / 'src' / 'outputs'
        outputs_dir.mkdir(parents=True, exist_ok=True)

        save_name = Path(save_csv).name if isinstance(save_csv, (str, Path)) else f"results.csv"
        out_path = outputs_dir / save_name

        df.to_csv(out_path, index=False, encoding="utf-8", sep=';')
        if verbose:
            print(f"\n✓ Results saved to: {out_path}")
    
    return df


def get_combined_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper: Merge GLiNER and regex columns intelligently.
    
    For columns that exist in both sources (e.g., 'email'),
    prefer GLiNER output but fallback to regex if empty.
    
    Args:
        df: DataFrame from process_single_cv()
    
    Returns:
        DataFrame with merged columns
    """
    if df.empty:
        return df
    
    # Create merged DataFrame
    merged = df.copy()
    
    # Define column mappings: (gliner_col, regex_col, merged_col)
    mappings = [
        ("person_name", "person_name_regex", "person_name_final"),
        ("job_title", None, "job_title_final"),  # Only GLiNER
        ("skills", None, "skills_final"),  # Only GLiNER
        ("company", None, "company_final"),  # Only GLiNER
        ("university", None, "university_final"),  # Only GLiNER
        ("degree", None, "degree_final"),  # Only GLiNER
    ]
    
    for gliner_col, regex_col, merged_col in mappings:
        if gliner_col in merged.columns:
            if regex_col and regex_col in merged.columns:
                # Prefer GLiNER, fallback to regex
                merged[merged_col] = merged[gliner_col].fillna(merged[regex_col])
            else:
                merged[merged_col] = merged[gliner_col]
    
    return merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        save_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        df = process_single_cv(pdf_path, save_csv=save_path, verbose=True)
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df.to_string())
    else:
        print("Usage: python process_single_cv.py <pdf_path> [output_csv_path]")
