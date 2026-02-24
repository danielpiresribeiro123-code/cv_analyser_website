from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from services.paths_models_and_data import get_gliner_model, SKILL_LABELS
from gliner import GLiNER

MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# Helpers: query / parsing
# -----------------------------
def split_items(x: str) -> List[str]:
    """
    Divide por vírgula/ponto-e-vírgula e limpa.
    Útil para queries de skills e também para o campo skills do CSV.
    """
    if not isinstance(x, str):
        return []
    s = x.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


# -----------------------------
# Degree inference (highest)
# -----------------------------
DEGREE_LEVELS = {
    "none": 0,
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "doctorate": 5,
}

# padrões comuns PT/EN (podes acrescentar)
DEGREE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(phd|ph\.d|doctorate|doutoramento|doutorado)\b", "doctorate"),
    (r"\b(masters|master|m\.sc|msc|mestrado)\b", "master"),
    (r"\b(bachelors|bachelor|b\.sc|bsc|licenciatura)\b", "bachelor"),
    (r"\b(associate)\b", "associate"),
    (r"\b(secondary|high\s*school|12º|12o|secund[aá]rio)\b", "highschool"),
]


def infer_highest_degree(degree_field: Any) -> Tuple[str, int]:
    """
    Recebe o campo 'degree' (string com possivelmente vários graus)
    e devolve (label, level) do grau mais alto encontrado.
    """
    s = normalize_text(str(degree_field) if degree_field is not None else "")
    s_low = s.lower()

    best_label = "none"
    best_level = DEGREE_LEVELS[best_label]

    for pattern, label in DEGREE_PATTERNS:
        if re.search(pattern, s_low):
            lvl = DEGREE_LEVELS[label]
            if lvl > best_level:
                best_level = lvl
                best_label = label

    return best_label, best_level


# -----------------------------
# Scoring functions
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def embed_mean(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embedding médio de uma lista de textos (ignora vazios).
    """
    texts = [normalize_text(t) for t in texts if normalize_text(t)]
    if not texts:
        # fallback: embedding de string vazia
        return model.encode([""], batch_size=1, show_progress_bar=False)[0]
    embs = model.encode(texts, batch_size=min(batch_size, len(texts)), show_progress_bar=False)
    return np.mean(embs, axis=0)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)




# -----------------------------
# Experience Scoring
# -----------------------------
def experience_log_scale(exp_years: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic scaling to experience years.

    This increases sensitivity at low experience levels
    and compresses differences at high experience levels.
    """
    exp_years = np.maximum(exp_years, 0.0)

    # log(1 + x) avoids log(0)
    log_values = np.log1p(exp_years)

    max_log = np.max(log_values)
    if max_log < 1e-9:
        return np.zeros_like(log_values)

    return log_values / max_log



# -----------------------------
# Gliner to extract skills from query text
# -----------------------------
def extract_skills_from_query(
    query_text: str,
    gliner_model: Optional[GLiNER] = None,
    threshold: float = 0.3
) -> List[str]:
    """
    Extract skill entities from query_text using GLiNER.
    """

    if not query_text.strip():
        return []

    if gliner_model is None:
        gliner_model = get_gliner_model()

    try:
        entities = gliner_model.predict_entities(query_text,SKILL_LABELS,threshold=threshold) # type: ignore[attr-defined]
    except Exception:
        return []

    skills = []
    for ent in entities:
        text = ent["text"].strip()
        if len(text) > 2:
            skills.append(text)

    # Remove duplicates ignoring case
    unique = {}
    for s in skills:
        key = s.lower()
        if key not in unique:
            unique[key] = s

    return list(unique.values())







def extract_job_titles_from_query(
    query_text: str,
    gliner_model: Optional[GLiNER] = None,
    threshold: float = 0.3
) -> List[str]:

    if not query_text.strip():
        return []

    if gliner_model is None:
        gliner_model = get_gliner_model()

    try:
        entities = gliner_model.predict_entities(query_text,["job title", "professional role", "occupation"],threshold=threshold) #type: ignore
    except Exception:
        return []

    titles = []
    for ent in entities:
        text = ent["text"].strip()
        if len(text) > 2:
            titles.append(text)

    # remove duplicates ignoring case
    unique = {}
    for t in titles:
        key = t.lower()
        if key not in unique:
            unique[key] = t

    return list(unique.values())