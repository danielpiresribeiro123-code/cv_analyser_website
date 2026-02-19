"""
Centralized paths and model loaders.

This module:
• Defines all backend paths
• Lazily loads heavy ML models (GLiNER, YOLO)
• Guarantees single instance per process (thread-safe)
• Prevents repeated disk/model loading
"""

from pathlib import Path
from typing import Optional, Any
from threading import Lock

from gliner import GLiNER
from ultralytics import YOLO  # type: ignore
import pytesseract

# =========================================================
# PATHS
# =========================================================

# Backend root
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Data folders
UPLOADS_DIR = BACKEND_ROOT / "uploads"
OUTPUTS_DIR = BACKEND_ROOT / "outputs"

# Model paths
GLINER_MODEL_PATH = BACKEND_ROOT / "models" / "gliner"
FILTER_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11n-text.pt"
TEXT_PHOTO_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11s-text-photo-150e4b.pt"
SKILL_TABLE_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11s-skill_table-150e4b.pt"


# =========================================================
# GLOBAL MODEL CACHE (thread-safe singletons)
# =========================================================

_GLINER_MODEL: Optional[Any] = None
_FILTER_YOLO_MODEL: Optional[YOLO] = None
_TEXT_PHOTO_YOLO_MODEL: Optional[YOLO] = None
_SKILL_TABLE_YOLO_MODEL: Optional[YOLO] = None

_GLINER_LOCK = Lock()
_FILTER_LOCK = Lock()
_TEXT_PHOTO_LOCK = Lock()
_SKILL_TABLE_LOCK = Lock()


# =========================================================
# GLiNER
# =========================================================

def get_gliner_model() -> GLiNER:
    """Return cached GLiNER model (loaded once)."""
    global _GLINER_MODEL

    if _GLINER_MODEL is None:
        with _GLINER_LOCK:
            if _GLINER_MODEL is None:
                if not GLINER_MODEL_PATH.exists():
                    raise FileNotFoundError(f"GLiNER model not found: {GLINER_MODEL_PATH}")

                _GLINER_MODEL = GLiNER.from_pretrained(str(GLINER_MODEL_PATH))

    return _GLINER_MODEL


# =========================================================
# YOLO — CV FILTER
# =========================================================

def get_filter_yolo_model() -> YOLO:
    """Return cached YOLO model for CV detection."""
    global _FILTER_YOLO_MODEL

    if _FILTER_YOLO_MODEL is None:
        with _FILTER_LOCK:
            if _FILTER_YOLO_MODEL is None:
                if not FILTER_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO filter model not found: {FILTER_YOLO_MODEL_PATH}")

                _FILTER_YOLO_MODEL = YOLO(str(FILTER_YOLO_MODEL_PATH))

    return _FILTER_YOLO_MODEL


# =========================================================
# YOLO — TEXT / PHOTO
# =========================================================

def get_text_photo_yolo_model() -> YOLO:
    """Return cached YOLO model for text/photo detection."""
    global _TEXT_PHOTO_YOLO_MODEL

    if _TEXT_PHOTO_YOLO_MODEL is None:
        with _TEXT_PHOTO_LOCK:
            if _TEXT_PHOTO_YOLO_MODEL is None:
                if not TEXT_PHOTO_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO text/photo model not found: {TEXT_PHOTO_YOLO_MODEL_PATH}")

                _TEXT_PHOTO_YOLO_MODEL = YOLO(str(TEXT_PHOTO_YOLO_MODEL_PATH))

    return _TEXT_PHOTO_YOLO_MODEL


# =========================================================
# YOLO — SKILL TABLE
# =========================================================

def get_skill_table_yolo_model() -> YOLO:
    """Return cached YOLO model for skill table detection."""
    global _SKILL_TABLE_YOLO_MODEL

    if _SKILL_TABLE_YOLO_MODEL is None:
        with _SKILL_TABLE_LOCK:
            if _SKILL_TABLE_YOLO_MODEL is None:
                if not SKILL_TABLE_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO skill table model not found: {SKILL_TABLE_YOLO_MODEL_PATH}")

                _SKILL_TABLE_YOLO_MODEL = YOLO(str(SKILL_TABLE_YOLO_MODEL_PATH))

    return _SKILL_TABLE_YOLO_MODEL


# ---------------------------------------------------------
# TESSERACT CONFIGURATION
# ---------------------------------------------------------

# Default Windows installation path, if you have custom chose custom
TESSERACT_EXE_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

_TESSERACT_CONFIGURED = False
_TESSERACT_LOCK = Lock()

def ensure_tesseract_configured() -> None:
    """
    Configure Tesseract executable path once per process.

    Unlike YOLO or GLiNER, Tesseract is NOT a Python model in memory.
    It is an external executable called on each OCR request.

    This function only ensures the executable path is set.
    """
    global _TESSERACT_CONFIGURED

    if not _TESSERACT_CONFIGURED:
        with _TESSERACT_LOCK:
            if not _TESSERACT_CONFIGURED:

                if not TESSERACT_EXE_PATH.exists():
                    raise FileNotFoundError(
                        f"Tesseract executable not found at: {TESSERACT_EXE_PATH}\n"
                        "Install Tesseract OCR or update TESSERACT_EXE_PATH."
                    )

                pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE_PATH)
                _TESSERACT_CONFIGURED = True

def run_tesseract_ocr(image, lang: str = "eng", config: str = "--oem 3 --psm 6") -> str:
    """
    Execute OCR using the configured Tesseract engine.

    Parameters
    ----------
    image : PIL.Image | numpy.ndarray
    lang : str
        OCR language code.
    config : str
        Tesseract configuration string.

    Returns
    -------
    str
        Extracted text.
    """
    ensure_tesseract_configured()

    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=config,
    )

    return text.strip()
