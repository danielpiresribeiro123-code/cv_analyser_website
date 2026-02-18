from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
import shutil

# =========================================================
# Local imports
# =========================================================
from backend.pipeline.pipelines.process_single_cv import process_single_cv #type: ignore

# =========================================================
# Project backend root resolution
# Current file path:
# backend\services\process_cvs_pipeline.py
#
# parents[0] -> services
# parents[1] -> backend
# =========================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def process_folder_cvs(
    input_folder: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    scan_subfolders: bool = False,
    verbose: bool = True,
    clear_output_folder: bool = False,
    crop_text_block:bool= False,
    crop_photo: bool = True,
    crop_skill_table: bool = False,
    save_ocr_txt: bool = False,
    **kwargs,) -> pd.DataFrame:
    """
    Process all CV files (PDF or image) inside a folder and return a single DataFrame.

    This function:
        • scans a directory for PDFs/images
        • runs process_single_cv() on each file
        • concatenates all results
        • optionally saves ONE global CSV

    Args:
        input_folder:
            Folder containing CV files.

        output_folder:
            Path to save the merged CSV.
            If None → does not save.

        scan_subfolders:
            If True, scans subfolders scan_subfoldersly.

        verbose:
            Prints progress.

        **kwargs:
            Extra parameters forwarded to process_single_cv()
            (confidence thresholds, cropping flags, etc.)

    Returns:
        pandas.DataFrame with all processed CV pages.
    """
    print("Input folder:", input_folder)
    print("Output folder:", output_folder)
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # -----------------------------------------------------
    # Prepare output folder
    # -----------------------------------------------------
    if output_folder is not None:
        output_folder = Path(output_folder)

        # Create if missing
        output_folder.mkdir(parents=True, exist_ok=True)

        # Optional cleaning
        if clear_output_folder:
            if verbose:
                print("Clearing output folder...")

            for item in output_folder.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"⚠️ Could not delete {item}: {e}")

    # -----------------------------------------------------
    # Determine files to process
    # -----------------------------------------------------
    patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]

    files: List[Path] = []

    # CASE 1 — single file
    if input_path.is_file():
        files = [input_path]

    # CASE 2 — folder
    else:
        for pattern in patterns:
            if scan_subfolders:
                files.extend(input_path.rglob(pattern))
            else:
                files.extend(input_path.glob(pattern))

        if not files:
            if verbose:
                print("⚠️ No CV files found.")
            return pd.DataFrame()

    if verbose:
        print("=" * 60)
        print(f"FOUND {len(files)} FILE(S) TO PROCESS")
        print("=" * 60)

    # -----------------------------------------------------
    # Process each file
    # -----------------------------------------------------
    all_dfs = []

    for idx, file_path in enumerate(files, 1):

        if verbose:
            print(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")

        try:
            df = process_single_cv(
                pdf_input=file_path,
                save_path=output_folder,
                crop_text_block=crop_text_block,
                crop_photo=crop_photo,
                crop_skill_table = crop_skill_table,
                save_ocr_txt = save_ocr_txt,
                verbose=verbose,
                **kwargs,)

            if not df.empty:
                df.insert(0, "source_file", file_path.name)
                all_dfs.append(df)

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    # -----------------------------------------------------
    # Merge results
    # -----------------------------------------------------
    if not all_dfs:
        if verbose:
            print("\n⚠️ No valid CV data extracted.")
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)

    # -----------------------------------------------------
    # Save global CSV
    # -----------------------------------------------------
    if output_folder is not None:
        csv_path = output_folder / "all_results.csv"
        final_df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

        if verbose:
            print(f"\n✓ Global CSV saved to: {csv_path}")


    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ BATCH COMPLETE: {len(final_df)} CV page(s)")
        print("=" * 60)

    return final_df


if __name__ == "__main__":
    df = process_folder_cvs(
        input_folder=BACKEND_ROOT /"uploads",
        output_folder=BACKEND_ROOT/"outputs",
        scan_subfolders=True,
        clear_output_folder= True,
        crop_text_block= True,
        crop_photo= True,
        crop_skill_table = True,
        save_ocr_txt = True,
        cv_confidence_threshold=0.3,
        yolo_confidence=0.25,
    )
