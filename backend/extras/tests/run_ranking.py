from services.ranking_pipeline import rank_cvs

if __name__ == "__main__":

    # Full contextual description (semantic analysis over raw text)
    query_text = """
    Looking for a Machine Learning Engineer with strong Python skills,
    experience with scikit-learn and NumPy, and knowledge of Docker and AWS.
    Experience with data processing and model deployment is a plus.
    """

    ranked_df = rank_cvs(
        query_text=query_text,
        gliner_model=None,
        top_k=10,   # Return only top 10 candidates
        weights ={
            "semantic_raw": 0.20,
            "semantic_skills": 0.40,
            "exact_skills": 0.15,
            "semantic_job_title": 0.06,
            "experience": 0.08,
            "degree": 0.05,
            "language": 0.06,})

    print(ranked_df[[
        "source_file",
        "person_name",
        "experience_years",
        "highest_degree",
        "semantic_raw_score",
        "semantic_skills_score",
        "final_score",]])