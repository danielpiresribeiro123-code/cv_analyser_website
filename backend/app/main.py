from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from fastapi.responses import FileResponse

# importar a tua fast analysis
from backend.pipeline.run_fast_analysis import run_fast_analysis


# =========================================================
# Configuração base
# =========================================================
app = FastAPI(title="CV Analyzer API", version="1.0")

# permitir ligação do frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois podes restringir
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# Endpoint de saúde (teste rápido)
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# Upload de CVs
# =========================================================
@app.post("/upload-cvs")
async def upload_cvs(files: list[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        dest = UPLOAD_DIR / filename

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(filename)

    return {
        "message": "CVs uploaded successfully",
        "files": saved_files,
        "upload_dir": str(UPLOAD_DIR)
    }


# =========================================================
# Executar pipeline (FAST ou FULL no futuro)
# =========================================================
@app.post("/run-pipeline")
def run_pipeline(mode: str = Query("fast", enum=["fast", "full"])):
    """
    mode:
        fast → usa embeddings/classificação rápida
        full → pipeline completa (ainda não implementada)
    """

    if mode == "fast":
        result = run_fast_analysis(str(UPLOAD_DIR))
        return result

    if mode == "full":
        return {
            "mode": "full",
            "status": "not implemented yet"
        }



RESULTS_CSV = OUTPUT_DIR / "fast_results.csv"


@app.get("/download-results")
def download_results():
    if not RESULTS_CSV.exists():
        return {"error": "No results available yet"}
    return FileResponse(
        path=RESULTS_CSV,
        filename="cv_results.csv",
        media_type="text/csv",
    )



from backend.pipeline.compute_ranking import compute_semantic_ranking
from pydantic import BaseModel


class RankingRequest(BaseModel):
    job_description: str


@app.post("/rank-candidates")
def rank_candidates(data: RankingRequest):
    ranking = compute_semantic_ranking(str(UPLOAD_DIR), data.job_description)

    return {
        "num_cvs": len(ranking),
        "ranking": ranking
    }
