from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import pandas as pd
from pydantic import BaseModel
import numpy as np
import os
import zipfile
import tempfile
from fastapi import Request

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



# ---------------------------------------------------------
# Upload CVs
# ---------------------------------------------------------
@router.post("/upload")
async def upload_cvs(files: list[UploadFile] = File(...)):

    print("UPLOAD ENDPOINT CALLED")
    print("UPLOADS DIR:", UPLOADS_DIR.resolve())

    UPLOADS_DIR.mkdir(exist_ok=True)

    for file in files:
        if not file.filename:
            continue
        print("Receiving:", file.filename)

        content = await file.read()
        print("Size:", len(content))

        file_path = UPLOADS_DIR / Path(file.filename).name
        print("Saving to:", file_path.resolve())

        with open(file_path, "wb") as buffer:
            buffer.write(content)

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
        crop_text_block=False,
        crop_photo=True,
        crop_skill_table=True,
        save_ocr_txt=True,
        analyse_individual_pages=False,
        cv_confidence_threshold=0.3,
        cv_min_detections=5,
        yolo_confidence=0.25,
        gliner_threshold=0.3,
        save_csv_single=False,
        gliner_model=get_gliner_model(),
        debug_save_json = True,
        filter_yolo_model=get_filter_yolo_model(),
        text_photo_yolo_model=get_text_photo_yolo_model(),
        skill_table_yolo_model=get_skill_table_yolo_model(),
        verbose=False,)

    # Ensure JSON-serializable values
    df = clean_df_for_json(df)

    return {
        "num_cvs": len(df),
        "results": df.to_dict(orient="records"),
    }


# ---------------------------------------------------------
# Retrieve processed results
# ---------------------------------------------------------
@router.get("/results")
def get_results(request: Request):
    base_url = str(request.base_url).rstrip("/")
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
            r["download_gliner_json_url"] = None
            continue

        folder_path = Path(folder)

        # RAW TEXT
        raw_path = folder_path / "raw_text.txt"
        r["download_raw_url"] = (
            f"{base_url}/download-raw?folder={folder}"
            if raw_path.exists()
            else None)

        # PHOTO
        photos_dir = folder_path / "photos"
        photo_exists = False
        if photos_dir.exists():
            for f in photos_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                    photo_exists = True
                    break

        r["photo_url"] = (
            f"{base_url}/photo?folder={folder}"
            if photo_exists
            else None)

        r["download_photo_url"] = (
            f"{base_url}/download-photo?folder={folder}"
            if photo_exists
            else None)

        # SKILL TABLES
        tables_dir = folder_path / "skill_tables"
        tables_exist = tables_dir.exists() and any(tables_dir.iterdir())

        r["download_skill_tables_url"] = (
            f"{base_url}/download-skill-tables?folder={folder}"
            if tables_exist
            else None)

        # GLiNER DEBUG JSON
        json_path = folder_path / "gliner_thresholds.json"

        r["download_gliner_json_url"] = (
            f"{base_url}/download-gliner-json?folder={folder}"
            if json_path.exists()
            else None)
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
    folder_path = Path(folder).resolve()

    # if its really inside OUTPUTS_DIR
    if not str(folder_path).startswith(str(OUTPUTS_DIR.resolve())):
        return {"error": "Invalid folder path"}

    photos_dir = folder_path / "photos"

    if not photos_dir.exists():
        return {"error": "No photo folder"}

    for file in photos_dir.iterdir():
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return FileResponse(file)

    return {"error": "No photo found"}



# ---------------------------------------------------------
# Download text raw, photo and skill table
# ---------------------------------------------------------
@router.get("/download-raw")
def download_raw(folder: str):
    folder_path = Path(folder).resolve()

    # if its really inside OUTPUTS_DIR
    if not str(folder_path).startswith(str(OUTPUTS_DIR.resolve())):
        return {"error": "Invalid folder path"}

    raw_path = folder_path / "raw_text.txt"

    if not raw_path.exists():
        return {"error": "Raw text not found"}

    return FileResponse(
        raw_path,
        media_type="text/plain",
        filename="raw_text.txt",
    )

@router.get("/download-photo")
def download_photo(folder: str):
    folder_path = Path(folder).resolve()

    # if its really inside OUTPUTS_DIR
    if not str(folder_path).startswith(str(OUTPUTS_DIR.resolve())):
        return {"error": "Invalid folder path"}

    photos_dir = folder_path / "photos"

    if not photos_dir.exists():
        return {"error": "No photo folder"}

    for file in photos_dir.iterdir():
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            return FileResponse(
                file,
                media_type="application/octet-stream",
                filename=file.name,
            )

    return {"error": "No photo found"}


@router.get("/download-skill-tables")
def download_skill_tables(folder: str):
    folder_path = Path(folder).resolve()

    # if its really inside OUTPUTS_DIR
    if not str(folder_path).startswith(str(OUTPUTS_DIR.resolve())):
        return {"error": "Invalid folder path"}
    
    tables_dir = folder_path / "skill_tables"

    if not tables_dir.exists():
        return {"error": "No skill tables folder"}

    files = list(tables_dir.iterdir())
    if not files:
        return {"error": "No skill tables found"}

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

    with zipfile.ZipFile(temp_zip.name, "w") as zipf:
        for file in files:
            zipf.write(file, arcname=file.name)

    return FileResponse(
        temp_zip.name,
        media_type="application/zip",
        filename="skill_tables.zip",
    )

# ---------------------------------------------------------
# Download GLiNER debug JSON
# ---------------------------------------------------------
@router.get("/download-gliner-json")
def download_gliner_json(folder: str):
    folder_path = Path(folder).resolve()

    # if its really inside OUTPUTS_DIR
    if not str(folder_path).startswith(str(OUTPUTS_DIR.resolve())):
        return {"error": "Invalid folder path"}

    json_path = folder_path / "gliner_thresholds.json"

    if not json_path.exists():
        return {"error": "GLiNER debug JSON not found"}

    return FileResponse(
        json_path,
        media_type="application/json",
        filename="gliner_thresholds.json",
    )

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

    ranking_df = rank_cvs(request.job_description, gliner_model=get_gliner_model())

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