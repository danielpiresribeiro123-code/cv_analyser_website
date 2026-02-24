from pathlib import Path
from typing import Union, Optional, List, Any
import pandas as pd
import shutil
from ultralytics import YOLO  # type: ignore
import csv

# =========================================================
# Local imports
# =========================================================
from pipeline.pipelines.process_single_cv import process_single_cv

# =========================================================
# Project backend root resolution
# =========================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def process_folder_cvs(
    input_folder: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    scan_subfolders: bool = False,
    verbose: bool = False,
    clear_output_folder: bool = False,
    crop_text_block: bool = False,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
    save_ocr_txt: bool = False,
    analyse_individual_pages: bool = False,
    cv_confidence_threshold: float = 0.3,
    cv_min_detections: int = 5,
    yolo_confidence: float = 0.25,
    gliner_threshold: float = 0.3,
    debug_save_json: bool = False,
    save_csv_single: bool = False,
    gliner_model: Any | None = None,
    filter_yolo_model: YOLO | None = None,
    text_photo_yolo_model: YOLO | None = None,
    skill_table_yolo_model: YOLO | None = None,) -> pd.DataFrame:
    """
    Batch processing of Curriculum Vitae (CV) files stored in a directory.

    This function orchestrates the automated analysis of multiple CV documents
    (PDF or image formats) by sequentially invoking the single-document
    processing pipeline. The extracted structured information from each CV is
    aggregated into a unified pandas DataFrame and can optionally be persisted
    to a global CSV file for incremental dataset construction.

    Core capabilities
    -----------------
    • Supports both PDF and common image-based CV formats.
    • Recursively scans subdirectories when enabled.
    • Performs incremental processing by skipping files that were already
      analyzed in previous executions, preventing redundant computation.
    • Safely appends new results to an existing CSV without overwriting
      historical data.
    • Optionally clears previous outputs to ensure a clean execution context.
    • Preserves traceability by storing source filename and absolute path
      for each processed CV.
    • Compatible with both document-level and page-level analysis modes.

    Parameters
    ----------
    input_folder : str or Path
        Path to a directory containing CV files or to a single CV file.

    output_folder : str or Path, optional
        Directory where intermediate artifacts and the aggregated CSV file
        will be stored. If None, results are returned only in memory.

    scan_subfolders : bool, default False
        If True, the function searches recursively for CV files inside all
        nested subdirectories.

    verbose : bool, default True
        Enables detailed console logging describing each processing step.

    clear_output_folder : bool, default False
        When enabled, removes previous output files before starting a new run,
        effectively resetting incremental history.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the merged structured results from all newly
        processed CV files, or the previously stored CSV data if no new files
        were analyzed.

    Notes
    -----
    This function acts as the batch-level coordination layer of the CV
    processing system. Heavy document analysis (layout detection, OCR,
    entity extraction, and contact parsing) is delegated to the
    `process_single_cv` pipeline, ensuring modularity, maintainability,
    and scalability for large-scale recruitment or research workflows.
    """

    if verbose:
        print("Input folder:", input_folder)
        print("Output folder:", output_folder)

    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # -----------------------------------------------------
    # Prepare output folder
    # -----------------------------------------------------
    csv_path: Optional[Path] = None

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        csv_path = output_folder / "all_results.csv"

        if clear_output_folder:
            if verbose:
                print("Output folder cleared. Previous results removed.")

            for item in output_folder.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as exc:
                    print(f"Warning: could not delete {item}: {exc}")

    # -----------------------------------------------------
    # Discover CV files
    # -----------------------------------------------------
    patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]

    if input_path.is_file():
        files: List[Path] = [input_path.resolve()]
    else:
        discovered: List[Path] = []

        for pattern in patterns:
            if scan_subfolders:
                discovered.extend(input_path.rglob(pattern))
            else:
                discovered.extend(input_path.glob(pattern))

        # remove duplicates + enforce deterministic order
        files = sorted({f.resolve() for f in discovered})

        if not files:
            if verbose:
                print("No CV files found.")
            return pd.DataFrame()

    if verbose:
        print("=" * 60)
        print(f"FOUND {len(files)} FILE(S) TO PROCESS")
        print("=" * 60)


    # -----------------------------------------------------
    # Load already processed CVs (incremental mode)
    # -----------------------------------------------------
    already_processed: set[str] = set()

    if csv_path and csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path, sep=";")

            if "source_path" in old_df.columns:
                already_processed = set(old_df["source_path"].dropna().astype(str))

            elif "source_file" in old_df.columns:
                # fallback compatível com CSV antigo
                already_processed = set(old_df["source_file"].dropna().astype(str))

            if verbose:
                print(f"Previously processed CVs found: {len(already_processed)}")

        except Exception as exc:
            print(f"Warning: could not read existing CSV: {exc}")

    # -----------------------------------------------------
    # Skip previously processed files
    # -----------------------------------------------------
    original_count = len(files)
    files = [
    f for f in files
    if str(f.resolve()) not in already_processed]

    if verbose:
        skipped = original_count - len(files)
        print(f"Files skipped (already processed): {skipped}")

    # -----------------------------------------------------
    # Process remaining files
    # -----------------------------------------------------
    all_dfs: List[pd.DataFrame] = []

    for idx, file_path in enumerate(files, start=1):

        if verbose:
            print(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")

        try:
            df = process_single_cv(
                cv_input=file_path,
                save_path=output_folder,
                cv_confidence_threshold=cv_confidence_threshold,
                cv_min_detections=cv_min_detections,
                yolo_confidence= yolo_confidence,
                gliner_threshold=gliner_threshold,
                crop_text_block=crop_text_block,
                crop_photo=crop_photo,
                crop_skill_table=crop_skill_table,
                analyse_individual_pages=analyse_individual_pages,
                save_ocr_txt=save_ocr_txt,
                save_csv_single=save_csv_single,
                gliner_model=gliner_model,
                debug_save_json = debug_save_json,
                filter_yolo_model=filter_yolo_model,
                text_photo_yolo_model=text_photo_yolo_model,
                skill_table_yolo_model=skill_table_yolo_model,
                verbose=verbose,)

            if not df.empty:
                df.insert(0, "source_file", str(file_path.name))
                df.insert(1, "source_path", str(file_path.resolve()))
                all_dfs.append(df)

        except Exception as exc:
            print(f"Error processing {file_path.name}: {exc}")

    # -----------------------------------------------------
    # If nothing new processed → return existing CSV
    # -----------------------------------------------------
    if not all_dfs:
        if verbose:
            print("\nNo new CVs processed.")

        if csv_path and csv_path.exists():
            return pd.read_csv(csv_path, sep=";")

        return pd.DataFrame()

    # -----------------------------------------------------
    # Merge new results
    # -----------------------------------------------------
    final_df = pd.concat(all_dfs, ignore_index=True)

    # -----------------------------------------------------
    # Save / append global CSV
    # -----------------------------------------------------
    merged_df: Optional[pd.DataFrame] = None

    if csv_path is not None:

        if csv_path.exists():
            old_df = pd.read_csv(csv_path, sep=";")
            merged_df = pd.concat([old_df, final_df], ignore_index=True)

            # Safety check for legacy CSVs without page_num
            if analyse_individual_pages and "page_num" not in merged_df.columns:
                analyse_individual_pages = False

            subset_cols = (
                ["source_path", "page_num"]
                if analyse_individual_pages
                else ["source_path"]
            )

            merged_df = merged_df.drop_duplicates(subset=subset_cols, keep="last")
            merged_df.to_csv(csv_path, index=False, encoding="utf-8", sep=";", quoting=csv.QUOTE_MINIMAL)

            if verbose:
                print(f"\nCSV updated: {csv_path}")

        else:
            final_df.to_csv(csv_path, index=False, encoding="utf-8", sep=";", quoting=csv.QUOTE_MINIMAL)

            if verbose:
                print(f"\nCSV created: {csv_path}")

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        unit = "page(s)" if analyse_individual_pages else "CV(s)"
        print(f"BATCH COMPLETE: {len(final_df)} new {unit}")
        print("=" * 60)

    if csv_path:
        return merged_df if merged_df is not None else final_df
    return final_df

