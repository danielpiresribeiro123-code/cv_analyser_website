from pathlib import Path
import json
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional

# Define label sets globally for reuse
SKILL_LABELS = [

    # ---------------------------
    # Core generic category
    # ---------------------------
    "skill",

    # ---------------------------
    # Technical skills
    # ---------------------------
    "programming language",
    "software framework",
    "library",
    "database",
    "cloud platform",
    "devops tool",
    "operating system",
    "web technology",
    "data analysis tool",
    "machine learning algorithm",
    "version control system",
    "software tool",
    "office suite",
    "creative software",

    # ---------------------------
    # Professional / domain skills
    # ---------------------------
    "methodology",
    "project management tool",
    "management skill",
    "domain knowledge",

    # ---------------------------
    # Soft skills
    # ---------------------------
    "soft skill",
    "leadership skill",
    "personal attribute",
    "language",
]

SOFTWARE_AS_SKILL = {
    "canva",
    "solidworks",
    "microsoft",
    "excel",
    "word",
    "powerpoint",
    "blender",
    "spyder",
    "vscode",
    "visual studio code",
    "davinci resolve",
}


OTHER_LABELS = [
    "person name", "address", "years of experience", 
    "job title", "company", "university", "degree",
]

ALL_LABELS = SKILL_LABELS + OTHER_LABELS


INVALID_ENTITIES = {
    "person name": {
        "person", "first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies"
    },
    "job title": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies"
    },
    "skills": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies"
    },
}


def extract_entities_from_text(
    text: str,
    model: GLiNER|None = None,
    threshold: float = 0.3
) -> Dict[str, List[str]]:
    """
    Extract entities from a single text using GLiNER.
    
    Args:
        text: The CV text to process (should be preprocessed beforehand if needed)
        model: GLiNER model instance. If None, loads default model.
        threshold: Confidence threshold for entity detection
    
    Returns:
        Dictionary with entity labels as keys and lists of extracted entities as values
    """
    if model is None:
        try:
            model = GLiNER.from_pretrained("backend/models/gliner")
        
        except Exception as e:
            print(e)
    
    
    # Predict entities
    entities = model.predict_entities(text, ALL_LABELS, threshold=threshold) 
    
    # Organize entities by label
    result_labels = OTHER_LABELS + ["skills"]
    entities_by_label = {label: [] for label in result_labels}
    
    for entity in entities:
        raw_text = entity["text"]
        text_norm = raw_text.lower().strip()
        # ----------------------------------------------
        # Heuristic 0 — remove invalid generic tokens
        # ----------------------------------------------
        invalid_set = INVALID_ENTITIES.get(entity["label"], set())
        if text_norm in invalid_set:
            continue

        if len(text_norm) <= 2:
            continue

                # --------------------------------------------------
        # Heuristic 1 — software dictionary (partial match)
        # --------------------------------------------------
        if any(sw in text_norm for sw in SOFTWARE_AS_SKILL):
            entities_by_label["skills"].append(raw_text)
            continue

        # --------------------------------------------------
        # Heuristic 2 — GLiNER skill labels
        # --------------------------------------------------
        if entity["label"] in SKILL_LABELS:
            entities_by_label["skills"].append(raw_text)
            continue

        # --------------------------------------------------
        # Heuristic 3 — filter fake companies
        # If detected as company but looks like software → skill
        # --------------------------------------------------
        if entity["label"] == "company":
            if any(sw in text_norm for sw in SOFTWARE_AS_SKILL):
                entities_by_label["skills"].append(raw_text)
                continue

        # --------------------------------------------------
        # Default handling
        # --------------------------------------------------
        if entity["label"] in OTHER_LABELS:
            entities_by_label[entity["label"]].append(raw_text)

        
    # Remove duplicates while preserving order
    for label in entities_by_label:
        entities_by_label[label] = list(dict.fromkeys(entities_by_label[label]))
    
    return entities_by_label


def process_texts_gliner(
    texts: Union[List[str], Dict[str, str]],
    model: GLiNER|None = None,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Process multiple texts and extract entities.
    Returns a pandas DataFrame with one row per text.
    
    Args:
        texts: Either:
            - List of text strings
            - Dict mapping IDs to text strings
        model: GLiNER model. If None, loads default.
        threshold: Entity detection threshold
    
    Returns:
        DataFrame with columns: id, skills, person_name, job_title, company, 
                               university, degree, address, years_of_experience
    
    Example:
        >>> from text_GliNER_preprocessing import preprocess_ocr_for_gliner
        >>> texts = [preprocess_ocr_for_gliner("Email: john@example.com\nSkills: Python, Java"), ...]
        >>> df = process_texts_gliner(texts)
    """
    if model is None:
        try:
            model = GLiNER.from_pretrained("backend/models/gliner")
        
        except Exception as e:
            print(e)

    # Normalize input
    if isinstance(texts, list):
        text_items = [(i, text) for i, text in enumerate(texts)]
    elif isinstance(texts, dict):
        text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")
    
    rows = []
    
    for text_id, text in text_items:
        try:
            # Extract entities
            entities_dict = extract_entities_from_text(
                text,
                model=model,
                threshold=threshold
            )
            
            # Flatten lists into comma-separated strings for DataFrame
            row = {"id": text_id}
            for label in OTHER_LABELS + ["skills"]:
                entities = entities_dict.get(label, [])
                row[label] = ", ".join(entities) if entities else None
            
            rows.append(row)
        
        except Exception as e:
            rows.append({
                "id": text_id,
                "error": str(e)
            })
    
    df = pd.DataFrame(rows)
    return df

