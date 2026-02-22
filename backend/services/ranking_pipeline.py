from pathlib import Path
from typing import List, Dict
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
    safe_float,
    normalize_text,
    infer_highest_degree,
    embed_mean,
    cosine_sim,
    minmax_norm,
    experience_log_scale,
    extract_skills_from_query
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
    raw_filename: str = "raw_text.txt",
    skills_col: str = "skills",
    exp_col: str = "experience_years",
    degree_col: str = "degree",
    id_col: str = "source_file",
    name_col: str = "person name",
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
            "semantic_raw": 0.20,
            "semantic_skills": 0.40,
            "exact_skills": 0.15,
            "semantic_job_title": 0.06,
            "experience": 0.08,
            "degree": 0.05,
            "language": 0.06,
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
    # Load raw_text.txt from each CV folder
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
    # Semantic similarity (raw text)
    # -----------------------------------------------------
    raw_embs = model.encode(raw_texts, batch_size=batch_size, show_progress_bar=True)
    raw_sims = np.array(
        [cosine_sim(raw_embs[i], q_raw_emb) for i in range(len(raw_embs))],
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

    exact_skill_scores = []

    # Lowercase query skills for exact matching
    query_skills_lower = [s.lower() for s in query_skills_list]

    for text in raw_texts:
        text_lower = text.lower()

        matches = 0

        for skill in query_skills_lower:
            # exact word / phrase match
            if re.search(rf"\b{re.escape(skill)}\b", text_lower):
                matches += 1

        if query_skills_lower:
            exact_score = matches / len(query_skills_lower)
        else:
            exact_score = 0.0

        exact_skill_scores.append(exact_score)

    exact_skill_scores = np.array(exact_skill_scores, dtype=float)

    # -----------------------------------------------------
    # Semantic similarity (job title)
    # -----------------------------------------------------
    job_titles = [
        normalize_text(str(row.get("job title", "")))
        for _, row in d.iterrows()]

    # Encode non-empty titles
    job_embs = model.encode(
        [jt if jt else "" for jt in job_titles],
        batch_size=batch_size,
        show_progress_bar=False,)

    job_sims = np.array(
        [cosine_sim(job_embs[i], q_raw_emb) for i in range(len(job_embs))],
        dtype=float,)

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

    if not query_languages:
        language_scores = np.ones(len(d))
    else:
        query_lang_lower_set = {ql.lower() for ql in query_languages}
        for _, row in d.iterrows():
            details_raw = row.get("languages_details", "")
            score_values = []
            extra_levels = []

            try:
                details = ast.literal_eval(details_raw) if isinstance(details_raw, str) else []
            except Exception:
                details = []

            cv_lang_map = {}
            for item in details:
                lang_name = str(item.get("language", "")).lower()
                level = item.get("level", None)
                cv_lang_map[lang_name] = level.lower() if isinstance(level, str) else None

            # ---- Requested languages score ----
            for q_lang in query_languages:
                q_lang_lower = q_lang.lower()

                if q_lang_lower in cv_lang_map:
                    level = cv_lang_map[q_lang_lower]
                    score_values.append(LEVEL_MAPPING.get(level, 0.50))
                else:
                    score_values.append(0.0)

            requested_score = float(np.mean(score_values)) if score_values else 0.0

            # ---- Extra languages bonus ----
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
    raw_norm = (raw_sims + 1.0) / 2.0
    skills_norm = (skill_sims + 1.0) / 2.0
    job_norm = (job_sims + 1.0) / 2.0

    # Normalize weights
    weight_sum = sum(max(v, 0.0) for v in weights.values()) or 1.0
    w = {k: max(v, 0.0) / weight_sum for k, v in weights.items()}

    # Final weighted score
    final_score = (
        w.get("semantic_raw", 0) * raw_norm
        + w.get("semantic_skills", 0) * skills_norm
        + w.get("exact_skills", 0) * exact_skill_scores
        + w.get("semantic_job_title", 0) * job_norm
        + w.get("experience", 0) * exp_norm
        + w.get("degree", 0) * degree_norm
        + w.get("language", 0) * language_scores)

    # -----------------------------------------------------
    # Build output dataframe
    # -----------------------------------------------------
    def col_or_empty(column_name: str):
        return d[column_name] if column_name in d.columns else [""] * len(d)

    ranking_df = pd.DataFrame({
        "source_file": col_or_empty(id_col),
        "person_name": col_or_empty(name_col),
        "job_title": col_or_empty("job title"),
        "email": col_or_empty("email"),
        "linkedin": col_or_empty("linkedin"),
        "experience_years": exp_years,
        "highest_degree": degree_labels,
        "semantic_raw_score": raw_sims,
        "semantic_skills_score": skill_sims,
        "exact_skill_score": exact_skill_scores,
        "semantic_job_title_score": job_sims,
        "final_score": final_score,
        "cv_output_folder": d[folder_col].astype(str),
        "language_score": language_scores,})

    ranking_df = ranking_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        ranking_df = ranking_df.head(int(top_k)).copy()

    # Save ranking to outputs folder
    ranking_path = OUTPUTS_PATH / "ranking.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8")

    return ranking_df