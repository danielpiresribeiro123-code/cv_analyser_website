from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sklearn.metrics.pairwise import cosine_similarity
import re
from services.paths_models_and_data import LANGUAGES
import ast

# =========================================================
# Local imports
# =========================================================
from pipeline.ranking.compute_ranking import (
    MODEL_NAME,
    DEGREE_LEVELS,
    split_items,
    split_multi_value_field,
    safe_float,
    normalize_text,
    infer_highest_degree,
    embed_mean,
    cosine_sim,
    experience_log_scale,
    extract_skills_from_query,
    weighted_job_degree_company_score,
    binary_presence_score,
    saturating_log_score,
    _language_count_score,
    split_text_for_semantic_matching,
    topk_semantic_text_score,
)



# =========================================================
# Project backend root resolution
# =========================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = BACKEND_ROOT / "outputs"
RESULTS_CSV_PATH = OUTPUTS_PATH / "all_results.csv"


def rank_cvs(
    query_text: str,
    gliner_model: GLiNER | None = None,
    *,
    sep: str = ";",
    folder_col: str = "cv_output_folder",
    raw_filename: str = "gliner_input.txt",
    fallback_raw_filename: str = "raw_text.txt",
    skills_col: str = "skills",
    exp_col: str = "experience_years",
    degree_col: str = "degree",
    id_col: str = "source_file",
    name_col: str = "person_name",
    ocr_len_col: str = "ocr_text_length",
    text_blocks_col: str = "num_text_blocks",
    photos_col: str = "num_photos",
    batch_size: int = 64,
    top_k: int | None = None,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute a hybrid CV ranking using:

    - Semantic similarity between query_text and raw_text.txt
    - Semantic similarity between query_skills and CSV skills column
    - Years of experience (min-max normalized)
    - Highest academic degree level (normalized)

    The final ranking is saved as OUTPUTS_PATH / "ranking.csv".
    """
    # -----------------------------------------------------
    # Validate query_text
    # -----------------------------------------------------
    if query_text is None or not str(query_text).strip():
        raise ValueError("query_text is empty. Please provide a non-empty query description.")

    if weights is None:
        weights = {
            "semantic_raw": 0.27,
            "semantic_skills": 0.30,
            "exact_skills": 0.03,
            "semantic_job_title": 0.08,
            "experience_general": 0.02,
            "experience_specific": 0.14,
            "degree": 0.05,
            "language": 0.06,
            "document_length": 0.03,
            "photo": 0.02,
        }

    # Load main CSV from outputs folder
    df = pd.read_csv(RESULTS_CSV_PATH, sep=sep, encoding="utf-8")

    for col in [folder_col, id_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    # -----------------------------------------------------
    # Extract skills from query using GLiNER
    # -----------------------------------------------------
    query_text = normalize_text(query_text)

    # -----------------------------------------------------
    # Extract languages from query_text
    # -----------------------------------------------------

    query_text_lower = query_text.lower()
    query_languages = []

    for lang in LANGUAGES:
        lang_lower = lang.lower()
        if re.search(rf"\b{re.escape(lang_lower)}\b", query_text_lower):
            query_languages.append(lang)

    # Remove detected languages from query_text
    for lang in query_languages:
        query_text = re.sub(
            rf"\b{re.escape(lang)}\b",
            "",
            query_text,
            flags=re.IGNORECASE
        )

    query_text = normalize_text(query_text)

    query_skills_list = extract_skills_from_query(
        query_text,
        gliner_model=gliner_model
    )

    # Fallback: if nothing detected, use keyword splitting
    if not query_skills_list:
        query_skills_list = split_items(query_text)
    if not query_skills_list:
        query_skills_list = [query_text]

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Compute query embeddings
    q_raw_emb = model.encode([query_text], show_progress_bar=False)[0]
    q_skills_emb = embed_mean(model, query_skills_list, batch_size=batch_size)

    # -----------------------------------------------------
    # Load gliner_input.txt or raw_text.txt from each CV folder
    # -----------------------------------------------------
    raw_texts = []
    valid_indices = []

    for i, row in df.iterrows():
        folder_value = row.get(folder_col, "")

        if not folder_value:
            continue

        folder_path = Path(folder_value)

        # If path is relative, assume it lives inside OUTPUTS_PATH
        if not folder_path.is_absolute():
            folder_path = OUTPUTS_PATH / folder_path

        raw_path = folder_path / raw_filename

        if not raw_path.exists():
            raw_path = folder_path / fallback_raw_filename

        if not raw_path.exists():
            continue

        text = raw_path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(text)

        if not text:
            continue

        raw_texts.append(text)
        valid_indices.append(i)

    if not valid_indices:
        raise RuntimeError("No raw_text.txt files could be loaded from CV folders.")

    d = df.loc[valid_indices].copy()

    # -----------------------------------------------------
    # Semantic similarity (raw text / top-k chunks)
    # -----------------------------------------------------
    raw_sims = np.array(
        [
            topk_semantic_text_score(
                model=model,
                query_emb=q_raw_emb,
                text=text,
                batch_size=batch_size,
                top_k=3,
                chunk_size=600,
                overlap=50,
                min_chunk_len=120,
                default_score=0.0,
            )
            for text in raw_texts
        ],
        dtype=float,
    )

    # -----------------------------------------------------
    # Semantic similarity (skills column) - Top-K strategy
    # -----------------------------------------------------

    TOP_K_SKILLS = 5  # configurable

    # Collect all skills across all CVs
    all_cv_skills = []
    cv_skill_counts = []  # number of skills per CV

    

    for _, row in d.iterrows():
        skills_list = split_items(str(row.get(skills_col, "")))
        cv_skill_counts.append(len(skills_list))
        all_cv_skills.extend(skills_list)

    # If no skills at all
    if not all_cv_skills:
        skill_sims = np.zeros(len(d))
    else:
        # Encode all skills in batch
        all_skill_embs = model.encode(
            all_cv_skills,
            batch_size=batch_size,
            show_progress_bar=False
        )

        # Compute similarity of each skill to query skills embedding
        similarities = cosine_similarity(all_skill_embs, q_skills_emb.reshape(1, -1)).flatten()

        # Now split similarities per CV and compute Top-K average
        skill_sims = []
        idx = 0

        for count in cv_skill_counts:
            if count == 0:
                skill_sims.append(0.0)
                continue

            cv_sims = similarities[idx: idx + count]
            idx += count

            # Select top K (or fewer if CV has less)
            k = min(TOP_K_SKILLS, len(cv_sims))

            # Sort descending
            top_k_values = np.sort(cv_sims)[-k:]

            mean_top_k = float(np.mean(top_k_values))
            max_similarity = float(np.max(cv_sims))

            # Hybrid scoring (more robust)
            score = 0.7 * mean_top_k + 0.3 * max_similarity

            skill_sims.append(score)

        skill_sims = np.array(skill_sims, dtype=float)

    # -----------------------------------------------------
    # Exact skill matching (symbolic)
    # -----------------------------------------------------

    def _normalize_skill_token(s: str) -> str:
        """
        Normalize a skill token for matching.
        Keeps symbols but normalizes whitespace and common spacing variants.
        """
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)

        # Common variants: "c ++" / "c + +" -> "c++"
        s = s.replace("c + +", "c++").replace("c ++", "c++")

        return s


    def _compile_skill_pattern(skill: str) -> re.Pattern:
        """
        Build a robust regex for exact matching.
        - If skill starts/ends with a word char, \b is usually safe.
        - If skill starts/ends with a non-word char (e.g., C++, C#, .NET),
        use alnum boundaries via lookarounds.
        """
        sk = _normalize_skill_token(skill)
        esc = re.escape(sk)

        # If the skill does NOT start and end with a word char, \b boundaries will be unreliable.
        # Examples: "c++", "c#", ".net"
        starts_word = bool(re.match(r"^\w", sk))
        ends_word = bool(re.match(r".*\w$", sk))

        if starts_word and ends_word:
            return re.compile(rf"\b{esc}\b", re.IGNORECASE)

        # Use "not alnum/underscore" boundaries instead of \b
        return re.compile(rf"(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])", re.IGNORECASE)


    exact_skill_scores = []

    # Lowercase/normalized query skills for exact matching
    query_skills_norm = [_normalize_skill_token(s) for s in query_skills_list if isinstance(s, str) and s.strip()]

    # Precompile patterns once (faster and consistent)
    query_skill_patterns = [_compile_skill_pattern(s) for s in query_skills_norm]

    for text in raw_texts:
        text_lower = (text or "").lower()

        matches = 0
        for pat in query_skill_patterns:
            if pat.search(text_lower):
                matches += 1

        exact_score = matches / len(query_skill_patterns) if query_skill_patterns else 0.0
        exact_skill_scores.append(exact_score)

    exact_skill_scores = np.array(exact_skill_scores, dtype=float)

    # -----------------------------------------------------
    # Semantic similarity (job titles + degrees + companies)
    # -----------------------------------------------------
    DEFAULT_EMPTY_JOB_SCORE = 0.2

    job_norm = []

    for _, row in d.iterrows():
        job_titles = split_multi_value_field(row.get("job_title", ""))
        degrees = split_multi_value_field(row.get(degree_col, ""))
        companies = split_multi_value_field(row.get("company", ""))

        score = weighted_job_degree_company_score(
            model=model,
            query_emb=q_raw_emb,
            job_titles=job_titles,
            degrees=degrees,
            companies=companies,
            batch_size=batch_size,
            default_score=DEFAULT_EMPTY_JOB_SCORE,
            job_weight=0.85,
            degree_weight=0.12,
            company_weight=0.03,
            fraction_divisor=5,
            min_top_n=1,
            max_top_n=5,   # or 6 if u want to limit it
            max_companies=4,
        )

        job_norm.append(score)

    job_norm = np.array(job_norm, dtype=float)

    # -----------------------------------------------------
    # Experience normalization
    # -----------------------------------------------------
    exp_years = np.array(
        [safe_float(x, 0.0) for x in d.get(exp_col, pd.Series([0] * len(d)))],
        dtype=float,
    )
    exp_norm = experience_log_scale(exp_years)

    # -----------------------------------------------------
    # Degree normalization
    # -----------------------------------------------------
    degree_labels = []
    degree_levels = []

    for value in d.get(degree_col, pd.Series([""] * len(d))):
        label, level = infer_highest_degree(value)
        degree_labels.append(label)
        degree_levels.append(level)

    degree_levels = np.array(degree_levels, dtype=float)
    degree_norm = degree_levels / float(max(DEGREE_LEVELS.values()))

    # -----------------------------------------------------
    # Document length scoring
    # -----------------------------------------------------
    ocr_lengths = np.array(
        [safe_float(x, 0.0) for x in d.get(ocr_len_col, pd.Series([0] * len(d)))],
        dtype=float,
    )

    text_blocks = np.array(
        [safe_float(x, 0.0) for x in d.get(text_blocks_col, pd.Series([0] * len(d)))],
        dtype=float,
    )

    # Approximate "useful length" saturation:
    # fast gains for short CVs, diminishing gains for long CVs.
    ocr_len_score = saturating_log_score(ocr_lengths, reference_value=2500)
    text_blocks_score = saturating_log_score(text_blocks, reference_value=40)

    document_length_score = 0.8 * ocr_len_score + 0.2 * text_blocks_score

    # -----------------------------------------------------
    # Photo scoring
    # -----------------------------------------------------
    num_photos = np.array(
        [safe_float(x, 0.0) for x in d.get(photos_col, pd.Series([0] * len(d)))],
        dtype=float,
    )

    photo_score = binary_presence_score(num_photos)

    # -----------------------------------------------------
    # Language scoring (with extra language bonus)
    # -----------------------------------------------------

    LEVEL_MAPPING = {
        "beginner": 0.25,
        "intermediate": 0.50,
        "advanced": 0.75,
        "native": 1.00,
        None: 0.50
    }

    language_scores = []

    query_lang_lower_set = {ql.lower() for ql in query_languages}

    for _, row in d.iterrows():
        details_raw = row.get("languages_details", "")
        score_values = []
        extra_levels = []

        try:
            if isinstance(details_raw, str):
                try:
                    details = ast.literal_eval(details_raw)
                except Exception:
                    details = []
            elif isinstance(details_raw, list):
                details = details_raw
            else:
                details = []
        except Exception:
            details = []

        cv_lang_map = {}
        for item in details:
            lang_name = str(item.get("language", "")).lower().strip()
            level = item.get("level", None)

            if lang_name:
                cv_lang_map[lang_name] = level.lower() if isinstance(level, str) else None

        # -------------------------------------------------
        # No languages explicitly requested in the query
        # -------------------------------------------------
        if not query_languages:
            language_scores.append(_language_count_score(len(cv_lang_map)))
            continue

        # -------------------------------------------------
        # Requested languages score
        # -------------------------------------------------
        for q_lang in query_languages:
            q_lang_lower = q_lang.lower()

            if q_lang_lower in cv_lang_map:
                level = cv_lang_map[q_lang_lower]
                score_values.append(LEVEL_MAPPING.get(level, 0.50))
            else:
                score_values.append(0.0)

        requested_score = float(np.mean(score_values)) if score_values else 0.0

        # -------------------------------------------------
        # Extra languages bonus
        # -------------------------------------------------
        for cv_lang, level in cv_lang_map.items():
            if cv_lang not in query_lang_lower_set:
                extra_levels.append(LEVEL_MAPPING.get(level, 0.50))

        if extra_levels:
            extra_bonus = min(0.25, float(np.mean(extra_levels)) * 0.25)
        else:
            extra_bonus = 0.0

        final_language_score = 0.75 * requested_score + (extra_bonus if requested_score > 0 else 0)
        language_scores.append(final_language_score)

    language_scores = np.array(language_scores, dtype=float)

    # -----------------------------------------------------
    # Normalize semantic scores to 0–1
    # -----------------------------------------------------
    # raw_norm = (raw_sims + 1.0) / 2.0
    raw_norm_v2 = np.clip(raw_sims / 0.6, 0.0, 1.0)
    skill_norm = np.clip(skill_sims / 0.85, 0.0, 1.0)
    job_sims = np.clip((job_norm - 0.5) / 0.4, 0.0, 1.0)
    exp_specific = np.clip((((job_sims + skill_norm)*exp_norm)/2) / 0.7, 0.0, 1.0)
    # Normalize weights
    weight_sum = sum(max(v, 0.0) for v in weights.values()) or 1.0
    w = {k: max(v, 0.0) / weight_sum for k, v in weights.items()}

    final_score = (
        w.get("semantic_raw", 0) * raw_norm_v2
        + w.get("semantic_skills", 0) * skill_norm
        + w.get("exact_skills", 0) * exact_skill_scores
        + w.get("semantic_job_title", 0) * job_sims
        + w.get("experience_general", 0) * exp_norm
        + w.get("experience_specific", 0) * exp_specific
        + w.get("degree", 0) * degree_norm
        + w.get("language", 0) * language_scores
        + w.get("document_length", 0) * document_length_score
        + w.get("photo", 0) * photo_score
    )

    # -----------------------------------------------------
    # Build output dataframe
    # -----------------------------------------------------
    def col_or_empty(column_name: str):
        return d[column_name] if column_name in d.columns else [""] * len(d)

    ranking_df = pd.DataFrame({
        "source_file": col_or_empty(id_col),
        "person_name": col_or_empty(name_col),
        "job_title": col_or_empty("job_title"),
        "email": col_or_empty("email"),
        "linkedin": col_or_empty("linkedin"),
        "experience_years": exp_years,
        "experience_score_general": exp_norm,
        "experience_score_specific": exp_specific,
        "highest_degree": degree_labels,
        "degree_score": degree_norm,
        "language_score": language_scores,
        "ocr_text_length": ocr_lengths,
        "num_text_blocks": text_blocks,
        "num_photos": num_photos,
        "document_length_score": document_length_score,
        "photo_score": photo_score,
        "semantic_raw_score": raw_norm_v2,
        "semantic_skills_score": skill_norm,
        "exact_skill_score": exact_skill_scores,
        "semantic_job_title_score": job_sims,
        "final_score": final_score,
        "cv_output_folder": d[folder_col].astype(str),
    })

    ranking_df = ranking_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        ranking_df = ranking_df.head(int(top_k)).copy()

    # Save ranking to outputs folder
    ranking_path = OUTPUTS_PATH / "ranking.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8")

    return ranking_df

