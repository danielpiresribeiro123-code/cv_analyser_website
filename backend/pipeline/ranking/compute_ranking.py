from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import math
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


DEGREE_PATTERNS: List[Tuple[str, str]] = [
    # -----------------------------
    # Doctorate (PhD etc)
    # -----------------------------
    (
        r"\b("
        r"phd|ph\.d|doctorate|doctoral|doctor\s+of\s+philosophy|"
        r"dphil|d\.phil|"
        r"doutoramento|doutorado|"
        r"doctorado|doctorat|doktorgrad|"
        r"jd|j\.d|juris\s+doctor"
        r")\b",
        "doctorate",
    ),
    # -----------------------------
    # Master
    # -----------------------------
    (
        r"\b("
        r"master|masters|master's|"
        r"m\.sc|msc|m\.s|ms|meng|m\.eng|ma|m\.a|mba|llm|"
        r"m\.tech|mtech|master\s+of\s+science|master\s+of\s+engineering|"
        r"master\s+of\s+arts|master\s+of\s+business\s+administration|"
        r"mestrado|mestre|mestrado\s+integrado|"
        r"postgraduate|post\s*graduate|"
        r"pós-graduação|pos-graduação|pos\s+graduacao|pós\s+graduação|"
        r"maestr[ií]a|mag[ií]ster|"
        r"mast[eè]re|"
        r"masterstudium|"
        r"laurea\s+magistrale|"
        r"yüksek\s+lisans"
        r")\b",
        "master",
    ),
    # -----------------------------
    # Bachelor
    # -----------------------------
    (
        r"\b("
        r"bachelor|bachelors|bachelor's|undergraduate|"
        r"b\.sc|bsc|b\.s|bs|beng|b\.eng|b\.e|be|"
        r"b\.tech|btech|ba|b\.a|bba|llb|"
        r"bachelor\s+of\s+science|bachelor\s+of\s+engineering|"
        r"bachelor\s+of\s+technology|bachelor\s+of\s+arts|"
        r"bachelor\s+of\s+business\s+administration|bachelor\s+of\s+laws|"
        r"licenciatura|licenciado|graduação|"
        r"grado|"
        r"laurea"
        r")\b",
        "bachelor",
    ),
    # -----------------------------
    # Medical doctor → treat as bachelor
    # -----------------------------
    (
        r"\b("
        r"doctor\b|dr\b|m\.d\b|md\b|doctor\s+of\s+medicine|medical\s+degree|"
        r"médico|medico|medicina"
        r")\b",
        "bachelor",
    ),
    # -----------------------------
    # Associate
    # -----------------------------
    (
        r"\b("
        r"associate|a\.a|aa|a\.s|as|associate\s+degree|associate\s+of\s+arts|associate\s+of\s+science|"
        r"cets|ctesp|curso\s+técnico\s+superior\s+profissional|"
        r"curso\s+tecnico\s+superior\s+profissional|"
        r"tecnico\s+superior"
        r")\b",
        "associate",
    ),
    # -----------------------------
    # High School
    # -----------------------------
    (
        r"\b("
        r"highschool|high\s*school|hs\s+diploma|secondary\s*school|"
        r"diploma\s+do\s+ensino\s+secundário|diploma\s+do\s+ensino\s+secundario|"
        r"ensino\s+secund[aá]rio|"
        r"12º|12o|12th\s+grade|12º\s+ano|12o\s+ano|12º\s+year|"
        r"bachillerato|lyc[eé]e"
        r")\b",
        "high school",
    ),
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
    threshold: float = 0.1
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





# -----------------------------
# More helpers
# -----------------------------

def split_multi_value_field(x: Any) -> List[str]:
    """
    Divide campos multi-valor típicos do CSV:
    - separados por vírgula
    - separados por ponto e vírgula
    - ignora vazios
    """
    if x is None:
        return []

    if isinstance(x, list):
        values = x
    else:
        s = normalize_text(str(x))
        if not s:
            return []
        s = s.replace(";", ",")
        values = [part.strip() for part in s.split(",")]

    out = []
    seen = set()

    for v in values:
        v_norm = normalize_text(str(v))
        if not v_norm:
            continue

        key = v_norm.lower()
        if key not in seen:
            seen.add(key)
            out.append(v_norm)

    return out


def best_semantic_match_score(
    model: SentenceTransformer,
    query_emb: np.ndarray,
    candidates: List[str],
    *,
    batch_size: int = 64,
    default_score: float = 0.2,
) -> float:
    """
    Calcula o melhor score semântico entre a query e uma lista de candidatos.
    Devolve score normalizado para [0, 1].

    Exemplo:
        candidates = ["Data Scientist", "Master of Science", "Python Developer"]
        -> devolve o maior score entre todos
    """
    clean_candidates = [normalize_text(c) for c in candidates if normalize_text(c)]

    if not clean_candidates:
        return default_score

    cand_embs = model.encode(
        clean_candidates,
        batch_size=min(batch_size, len(clean_candidates)),
        show_progress_bar=False
    )

    sims = cosine_similarity(cand_embs, query_emb.reshape(1, -1)).flatten()
    best_sim = float(np.max(sims))

    # cosine [-1, 1] -> [0, 1]
    return (best_sim + 1.0) / 2.0


def _language_count_score(num_languages: int) -> float:
    """
    Score based only on how many languages the CV contains
    when the query does not explicitly request languages.
    """
    if num_languages <= 0:
        return 0.0
    if num_languages == 1:
        return 0.50
    if num_languages == 2:
        return 0.75
    if num_languages == 3:
        return 0.90
    if num_languages == 4:
        return 0.95
    return 1.00

def split_text_for_semantic_matching(
    text: str,
    chunk_size: int = 500,
    overlap: int = 200,
    min_chunk_len: int = 120,
) -> List[str]:
    """
    Split text into fixed-size overlapping chunks.

    Strategy:
    - preserve semantic continuity with overlap
    - useful for CV ranking where relevant info may be localized
    - chunk_size and overlap are character-based
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    if chunk_size <= 0:
        return [text]

    if overlap < 0:
        overlap = 0

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)

    step = chunk_size - overlap
    chunks: List[str] = []

    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        # If final chunk is too short, merge with previous by taking a larger tail window
        if len(chunk) < min_chunk_len and chunks:
            tail_start = max(0, n - chunk_size)
            chunk = text[tail_start:n].strip()

            if chunk and chunk != chunks[-1]:
                chunks.append(chunk)
            break

        if chunk:
            if not chunks or chunk != chunks[-1]:
                chunks.append(chunk)

        if end >= n:
            break

        start += step

    return chunks

def topk_semantic_text_score(
    model: SentenceTransformer,
    query_emb: np.ndarray,
    text: str,
    *,
    batch_size: int = 64,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 200,
    min_chunk_len: int = 120,
    default_score: float = 0.0,
) -> float:
    """
    Compare query against overlapping text chunks and return a robust top-k semantic score.
    """
    chunks = split_text_for_semantic_matching(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_len=min_chunk_len,
    )

    if not chunks:
        return default_score

    embs = model.encode(
        chunks,
        batch_size=min(batch_size, len(chunks)),
        show_progress_bar=False,
    )

    sims = cosine_similarity(embs, query_emb.reshape(1, -1)).flatten()

    if len(sims) == 0:
        return default_score

    k = min(top_k, len(sims))
    top_vals = np.sort(sims)[-k:]

    # média dos melhores chunks
    return float(np.mean(top_vals))

def _topn_or_best_score(
    sims: np.ndarray,
    *,
    fraction_divisor: int = 4,
    min_top_n: int = 1,
    max_top_n: Optional[int] = None,
) -> Optional[float]:
    """
    Dynamic top-N rule based on ceil(num_items / fraction_divisor).

    Examples with fraction_divisor=4:
    - 3 items  -> top 1
    - 11 items -> top 3
    - 21 items -> top 6

    Formula:
        n = max(min_top_n, ceil(num_items / fraction_divisor))
    """
    if sims.size == 0:
        return None

    if fraction_divisor <= 0:
        fraction_divisor = 4

    n = max(min_top_n, math.ceil(sims.size / fraction_divisor))

    if max_top_n is not None:
        n = min(n, max_top_n)

    n = min(n, sims.size)

    top_vals = np.sort(sims)[-n:]
    return float(np.mean(top_vals))


def _company_count_score(companies: List[str], max_companies: int = 4) -> Optional[float]:
    """
    Company score based only on number of companies.
    Each company contributes equally up to max_companies.

    Examples:
    - 0 companies -> None
    - 1 company   -> 0.25
    - 2 companies -> 0.50
    - 3 companies -> 0.75
    - 4+ companies -> 1.00
    """
    clean_companies = [normalize_text(c) for c in companies if normalize_text(c)]

    if not clean_companies:
        return None

    count = min(len(clean_companies), max_companies)
    return float(count / max_companies)

def weighted_job_degree_company_score(
    model: SentenceTransformer,
    query_emb: np.ndarray,
    job_titles: List[str],
    degrees: List[str],
    companies: List[str],
    *,
    batch_size: int = 64,
    default_score: float = 0.2,
    job_weight: float = 0.85,
    degree_weight: float = 0.12,
    company_weight: float = 0.03,
    fraction_divisor: int = 4,
    min_top_n: int = 1,
    max_top_n: Optional[int] = None,
    max_companies: int = 4,
) -> float:
    """
    Combined score using custom fixed logic:

    - company is NEVER redistributed
    - if both job and degree exist:
        score = job_weight * job_score + degree_weight * degree_score + company_weight * company_score
    - if only job exists:
        score = (job_weight / 2) * job_score + company_weight * company_score
    - if only degree exists:
        score = (degree_weight / 2) * degree_score + company_weight * company_score
    - if neither job nor degree exists:
        score = default_score + company_weight * company_score

    For jobs/degrees:
    - use the mean of the best ceil(n/4) items
    - with minimum 1 item
    """

    clean_jobs = [normalize_text(x) for x in job_titles if normalize_text(x)]
    clean_degrees = [normalize_text(x) for x in degrees if normalize_text(x)]
    clean_companies = [normalize_text(x) for x in companies if normalize_text(x)]

    def _encode_and_score(candidates: List[str]) -> Optional[float]:
        if not candidates:
            return None

        embs = model.encode(
            candidates,
            batch_size=min(batch_size, len(candidates)),
            show_progress_bar=False,
        )
        sims = cosine_similarity(embs, query_emb.reshape(1, -1)).flatten()

        # cosine [-1, 1] -> [0, 1]
        sims = (sims + 1.0) / 2.0

        return _topn_or_best_score(
            sims,
            fraction_divisor=fraction_divisor,
            min_top_n=min_top_n,
            max_top_n=max_top_n,
        )

    job_score = _encode_and_score(clean_jobs)
    degree_score = _encode_and_score(clean_degrees)
    company_score = _company_count_score(clean_companies, max_companies=max_companies)

    company_part = company_weight * company_score if company_score is not None else 0.0

    # Both exist
    if job_score is not None and degree_score is not None:
        return float(
            job_weight * job_score
            + degree_weight * degree_score
            + company_part
        )

    # Only job exists
    if job_score is not None and degree_score is None:
        return float(
            (job_weight / 2.0) * job_score
            + company_part
        )

    # Only degree exists
    if degree_score is not None and job_score is None:
        return float(
            (degree_weight / 2.0) * degree_score
            + company_part
        )

    # Neither job nor degree exists
    return float(default_score + company_part)

# -----------------------------
# Helpers for lenght and photos
# -----------------------------
def robust_log_norm(values: np.ndarray) -> np.ndarray:
    """
    Log transform + min-max normalization.
    Good for skewed count-like variables such as OCR text length,
    number of blocks, number of pages, etc.
    """
    values = np.maximum(values.astype(float), 0.0)
    logged = np.log1p(values)

    mn = float(np.min(logged))
    mx = float(np.max(logged))

    if mx - mn < 1e-9:
        return np.zeros_like(logged, dtype=float)

    return (logged - mn) / (mx - mn)


def binary_presence_score(values: np.ndarray) -> np.ndarray:
    """
    Convert numeric counts into binary presence score:
    > 0 -> 1.0
    <= 0 -> 0.0
    """
    values = np.maximum(values.astype(float), 0.0)
    return (values > 0).astype(float)

def saturating_log_score(
    values: np.ndarray,
    reference_value: float,
) -> np.ndarray:
    """
    Log-based saturating score in [0, 1].

    - Grows quickly for small values
    - Grows slowly for large values
    - Reaches ~1.0 around reference_value
    - Anything above reference_value is clipped to 1.0
    """
    values = np.maximum(values.astype(float), 0.0)

    if reference_value <= 0:
        return np.zeros_like(values, dtype=float)

    scores = np.log1p(values) / np.log1p(reference_value)
    return np.clip(scores, 0.0, 1.0)