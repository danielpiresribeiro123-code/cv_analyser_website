from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import pandas as pd
from pydantic import BaseModel
import numpy as np
import os
from fastapi.responses import FileResponse

# =========================================================
# Internal services
# =========================================================
from services.process_cvs_pipeline import process_folder_cvs
from services.storage_cleanup import clear_uploads, clear_all_storage, clean_df_for_json
from services.ranking_pipeline import rank_cvs

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


# =========================================================
# Models
# =========================================================
gliner = get_gliner_model()

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
        clear_output_folder=False,   # preserve incremental history, we have a bottum that removes the data from the outputs
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
        gliner_model=gliner,
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

    csv_path = OUTPUTS_DIR / "all_results.csv"

    if not csv_path.exists():
        return {"num_cvs": 0, "results": []}

    df = pd.read_csv(csv_path, sep=";")
    df = clean_df_for_json(df)

    records = df.to_dict(orient="records")

    for r in records:
        folder = r.get("cv_output_folder")

        if not folder:
            r["photo_url"] = None
            r["download_raw_url"] = None
            r["download_photo_url"] = None
            r["download_skill_tables_url"] = None
            continue

        folder_path = Path(folder)

        # RAW TEXT
        raw_path = folder_path / "raw_text.txt"
        r["download_raw_url"] = (
            f"http://127.0.0.1:8000/download-raw?folder={folder}"
            if raw_path.exists()
            else None
        )

        # PHOTO
        photos_dir = folder_path / "photos"
        photo_exists = False
        if photos_dir.exists():
            for f in photos_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                    photo_exists = True
                    break

        r["photo_url"] = (
            f"http://127.0.0.1:8000/photo?folder={folder}"
            if photo_exists
            else None
        )

        r["download_photo_url"] = (
            f"http://127.0.0.1:8000/download-photo?folder={folder}"
            if photo_exists
            else None
        )

        # SKILL TABLES
        tables_dir = folder_path / "skill_tables"
        tables_exist = tables_dir.exists() and any(tables_dir.iterdir())

        r["download_skill_tables_url"] = (
            f"http://127.0.0.1:8000/download-skill-tables?folder={folder}"
            if tables_exist
            else None
        )

    return {
        "num_cvs": len(records),
        "results": records,
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



# ---------------------------------------------------------
# Serve CV photo
# ---------------------------------------------------------
@router.get("/photo")
def get_cv_photo(folder: str):
    """
    Returns first available photo inside CV output folder.
    """

    folder_path = Path(folder)

    photos_dir = folder_path / "photos"

    if not photos_dir.exists():
        return {"error": "No photo folder"}

    for file in photos_dir.iterdir():
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return FileResponse(file)

    return {"error": "No photo found"}



# ---------------------------------------------------------
# Job description and ranking of the results
# ---------------------------------------------------------
class RankingRequest(BaseModel):
    job_description: str


@router.post("/rank-candidates")
def rank_candidates(request: RankingRequest):
    """
    Run semantic ranking over processed CVs.
    """

    ranking_df = rank_cvs(request.job_description, gliner_model=gliner)

    # Converter para JSON-friendly
    ranking_df = ranking_df.replace({np.nan: None})

    return ranking_df.to_dict(orient="records")


# ---------------------------------------------------------
# Download Ranking CSV
# ---------------------------------------------------------
@router.get("/download-ranking")
def download_ranking_csv():
    """
    Download the generated ranking CSV.
    """

    csv_path = OUTPUTS_DIR / "ranking.csv"

    if not csv_path.exists():
        return {"error": "Ranking CSV not found"}

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="ranking.csv",
    )