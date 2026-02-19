from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import pandas as pd

# =========================================================
# Internal services
# =========================================================
from services.process_cvs_pipeline import process_folder_cvs
from services.storage_cleanup import clear_uploads, clear_all_storage, clean_df_for_json

# Centralized model + path loaders
from services.paths_models_and_data import (
    BACKEND_ROOT,
    UPLOADS_DIR,
    OUTPUTS_DIR,
    get_gliner_model,
    get_filter_yolo_model,
    get_text_photo_yolo_model,
    get_skill_table_yolo_model,)

router = APIRouter()

# Ensure folders exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Upload CVs
# ---------------------------------------------------------
@router.post("/upload")
async def upload_cvs(files: list[UploadFile] = File(...)):
    """
    Upload new CV files to the server.

    Behavior
    --------
    • Clears previous uploads to isolate processing sessions
    • Stores new files inside the backend uploads directory
    """

    # Reset previous upload session
    clear_uploads(UPLOADS_DIR)

    # Save incoming files
    for file in files:
        if not file.filename:
            continue

        file_path = UPLOADS_DIR / Path(file.filename).name

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return {"message": "Files uploaded successfully"}


# ---------------------------------------------------------
# Run processing pipeline
# ---------------------------------------------------------
@router.post("/process")
def run_pipeline():
    """
    Execute the full CV processing pipeline over uploaded files.

    Pipeline characteristics
    ------------------------
    • Uses cached ML models (GLiNER + YOLO) loaded once per process
    • Aggregates structured results into outputs/all_results.csv
    • Returns JSON-ready cleaned results for frontend consumption
    """

    df = process_folder_cvs(
        input_folder=UPLOADS_DIR,
        output_folder=OUTPUTS_DIR,
        scan_subfolders=False,
        clear_output_folder=False,   # preserve incremental history
        crop_text_block=True,
        crop_photo=True,
        crop_skill_table=True,
        save_ocr_txt=True,
        analyse_individual_pages=False,
        cv_confidence_threshold=0.3,
        cv_min_detections=5,
        yolo_confidence=0.25,
        gliner_threshold=0.3,
        save_csv=False,
        gliner_model=get_gliner_model(),
        filter_yolo_model=get_filter_yolo_model(),
        text_photo_yolo_model=get_text_photo_yolo_model(),
        skill_table_yolo_model=get_skill_table_yolo_model(),
        verbose=False,
    )

    # Ensure JSON-serializable values
    df = clean_df_for_json(df)

    return {
        "num_cvs": len(df),
        "results": df.to_dict(orient="records"),
    }


# ---------------------------------------------------------
# Retrieve processed results (JSON)
# ---------------------------------------------------------
@router.get("/results")
def get_results():
    """
    Return previously processed CV results stored in CSV.
    """

    csv_path = OUTPUTS_DIR / "all_results.csv"

    if not csv_path.exists():
        return {"num_cvs": 0, "results": []}

    df = pd.read_csv(csv_path, sep=";")
    df = clean_df_for_json(df)

    return {
        "num_cvs": len(df),
        "results": df.to_dict(orient="records"),
    }


# ---------------------------------------------------------
# Download CSV results
# ---------------------------------------------------------
@router.get("/download")
def download_csv():
    """
    Download the aggregated CSV containing all processed CV results.
    """

    csv_path = OUTPUTS_DIR / "all_results.csv"

    if not csv_path.exists():
        return {"error": "CSV not found"}

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="all_results.csv",
    )


# ---------------------------------------------------------
# Manual system reset
# ---------------------------------------------------------
@router.post("/reset")
def reset_system():
    """
    Clear uploads, outputs, and YOLO temporary run folders.

    Useful for:
    • Restarting frontend demo sessions
    • Cleaning incremental processing history
    """

    clear_all_storage(UPLOADS_DIR, OUTPUTS_DIR, BACKEND_ROOT)

    return {"message": "System reset successfully"}

