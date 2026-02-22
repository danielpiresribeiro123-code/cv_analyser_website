import re
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional, Any, cast

from services.paths_models_and_data import get_gliner_model, SKILL_LABELS, COUNTRIES_AND_CAPITALS, COUNTRIES_NAMES, CAPITALS_NAMES, LANGUAGES


# Define label sets globally for reuse

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

ENTITY_GROUPS = {
    "skills": SKILL_LABELS,
    "person name": ["person name"],
    "job title": ["job title"],
    "company": ["company", "corporation"],
    "university": ["university","school", "college", "faculty"],
    "degree": ["degree", "bachelor", "master", "doctorate"],
    "address": ["address","location","street", "city", "town", "country", "state", "region"],}

ALL_LABELS = list(dict.fromkeys(
    label
    for labels in ENTITY_GROUPS.values()
    for label in labels
))

INVALID_ENTITIES = {
    "person name": {
        "person", "first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies"},
    "job title": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background", "background","education", "experience", "hobbies"},
    "skills": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies", "C1", "First Certificate"},
    "degree": {"degree", "education"},
    "company": {"company", "organization"},
    "address": {"address", "location"}}

POSTAL_PATTERN = re.compile(r"\b\d{4,6}[- ]?\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\b(nº|no\.?|number)?\s?\d+\b", re.IGNORECASE)

NAME_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")

# Normalize language list once
LANGUAGES_SET = {lang.lower().strip() for lang in LANGUAGES}

def _is_valid_skill(text: str) -> bool:
    """
    Validate if a skill candidate is acceptable.

    Removes:
    - Languages (English, Portuguese, etc.)
    - Generic placeholders (academic background, education, etc.)
    - Very short meaningless tokens
    """

    if not text:
        return False

    t = text.lower().strip()

    # Remove languages (exact match)
    if t in LANGUAGES_SET:
        return False

    # Remove if contains language as full token
    for lang in LANGUAGES_SET:
        if re.search(rf"\b{re.escape(lang)}\b", t):
            return False

    # Remove generic non-skills
    GENERIC_BAD_SKILLS = {
        "academic background",
        "academic",
        "background",
        "education",
        "experience",
        "hobbies",
        "soft skills",
        "hard skills",
        "skills",
        "skill"
    }

    if t in GENERIC_BAD_SKILLS:
        return False

    # Remove single character tokens
    if len(t) <= 2:
        return False

    return True

def _clean_person_name(entities: List[str]) -> List[str]:
    """
    Clean and normalize person name entities.

    Strategy:
        1. Remove tokens containing digits or symbols.
        2. Prefer tokens in Proper Case (John Silva).
        3. If at least two Proper Case tokens exist, keep the first two.
        4. Otherwise, keep original uppercase full name (e.g., JOHN SILVA).
    """

    cleaned_results = []

    for raw in entities:
        tokens = raw.replace(",", " ").split()

        # Keep only alphabetic tokens
        tokens = [t for t in tokens if NAME_TOKEN_PATTERN.match(t)]

        if len(tokens) < 2:
            continue

        # Proper Case detection (John Silva)
        proper_tokens = [
            t for t in tokens
            if t[0].isupper() and t[1:].islower()
        ]

        if len(proper_tokens) >= 2:
            cleaned_results.append(" ".join(proper_tokens[:2]))
        else:
            # fallback: keep first two alphabetic tokens
            cleaned_results.append(" ".join(tokens[:2]))

    return list(dict.fromkeys(cleaned_results))

def _remove_substring_duplicates(values: List[str]) -> List[str]:
    """
    Remove entities that are strict substrings of longer entities.
    Keeps the most specific version of overlapping entities.

    Example:
        ["FCT NOVA", "NOVA"] → ["FCT NOVA"]
    """

    values_sorted = sorted(values, key=len, reverse=True)
    filtered = []

    for val in values_sorted:
        if not any(val != other and val.lower() in other.lower() for other in filtered):
            filtered.append(val)

    return filtered

def _choose_first_address(addresses: List[str]) -> List[str]:
    """
    Choose the most complete address.
    Priority:
        1. Longest address
        2. First if equal length
    """
    if not addresses:
        return []

    valid = [a.strip() for a in addresses if isinstance(a, str) and a.strip()]
    if not valid:
        return []

    # Escolhe o mais longo
    best = max(valid, key=len)

    return [best]

def _clean_degree_entities(values: List[str]) -> List[str]:
    """
    Keep full degree descriptions.
    Remove exact duplicates only.
    """
    return list(dict.fromkeys(values))

def _expand_address_smart(full_text: str, address_fragment: str, window: int = 80, capitals_names: set|None = COUNTRIES_NAMES, countries_names: set|None=CAPITALS_NAMES) -> str:
    """
    Expand an address fragment by searching nearby for:
        - Postal codes
        - Street numbers
        - Capitals
        - Country names

    Keeps the closest structural signals.
    """
    COUNTRIES = set(COUNTRIES_AND_CAPITALS .keys())
    CAPITALS = {v for v in COUNTRIES_AND_CAPITALS .values() if v}
    match = re.search(re.escape(address_fragment), full_text, re.IGNORECASE)
    if not match:
        return address_fragment

    start, end = match.span()

    left_bound = max(0, start - window)
    right_bound = min(len(full_text), end + window)

    expanded_start = start
    expanded_end = end

    # Search right side
    right_text = full_text[end:right_bound]
    right_text_lower = right_text.lower()

    postal = POSTAL_PATTERN.search(right_text)
    if postal:
        expanded_end = end + postal.end()

    capitals = capitals_names if capitals_names is not None else CAPITALS
    countries = countries_names if countries_names is not None else COUNTRIES

    for name in capitals.union(countries):
        name_lower = name.lower()
        idx = right_text_lower.find(name_lower)
        if idx != -1:
            expanded_end = max(expanded_end, end + idx + len(name))

    # Search left side
    left_text = full_text[left_bound:start]

    number_matches = list(NUMBER_PATTERN.finditer(left_text))
    if number_matches:
        expanded_start = left_bound + number_matches[-1].start()

    expanded = full_text[expanded_start:expanded_end]

    expanded = re.sub(r"\s+", " ", expanded)
    expanded = expanded.strip(" ,;")

    return expanded

def extract_entities_from_text(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.3
) -> Dict[str, List[str]]:

    result_labels = [
                "person name",
                "job title",
                "company",
                "university",
                "degree",
                "address",
                "skills"]

    if not text or not text.strip():
        return {label: [] for label in result_labels}

    if model is None:
        model = get_gliner_model()

    model = cast(GLiNER, model)

    try:
        entities = model.predict_entities(text, ALL_LABELS, threshold=threshold)  # type: ignore[attr-defined]
    except Exception:
        return {label: [] for label in result_labels}

    entities_by_label = {label: [] for label in result_labels}

    for entity in entities:
        raw_text = entity["text"]
        text_norm = raw_text.lower().strip()
        label = entity["label"]

        if len(text_norm) <= 2:
            continue

        # remove genéricos
        invalid_set = INVALID_ENTITIES.get(label, set())
        if text_norm in invalid_set:
            continue

        # skills
        if label in ENTITY_GROUPS["skills"]:
            if _is_valid_skill(raw_text):
                entities_by_label["skills"].append(raw_text)
            continue

        if any(re.search(rf"\b{re.escape(sw)}\b", text_norm) for sw in SOFTWARE_AS_SKILL):
            if _is_valid_skill(raw_text):
                entities_by_label["skills"].append(raw_text)
            continue
        
        # address
        if label in ENTITY_GROUPS["address"]:
            expanded = _expand_address_smart(text, raw_text)
            entities_by_label["address"].append(expanded)
            continue
      
        # outras labels diretas
        for group_name, group_labels in ENTITY_GROUPS.items():
            if group_name in ("skills", "address"):
                continue
            if label in group_labels:
                entities_by_label[group_name].append(raw_text)
                break

    return entities_by_label

def process_texts_gliner(
    texts: Union[List[str], Dict[str, str]],
    model: Optional[GLiNER] = None,
    threshold: float = 0.3
) -> pd.DataFrame:

    if model is None:
        model = get_gliner_model()

    if isinstance(texts, list):
        text_items = list(enumerate(texts))
    elif isinstance(texts, dict):
        text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")

    rows = []

    for _, text in text_items:
        try:
            entities_dict = extract_entities_from_text(
                text,
                model=model,
                threshold=threshold
            )

            row: Dict[str, Any] = {}

            for label in [
                "person name",
                "job title",
                "company",
                "university",
                "degree",
                "address",
                "skills"
            ]:

                entities = entities_dict.get(label, [])

                clean_entities = []
                for e in entities:
                    if not isinstance(e, str):
                        continue

                    e_clean = " ".join(e.replace("\n", " ").replace("\r", " ").split())
                    if e_clean:
                        clean_entities.append(e_clean.strip())

                if label == "person name":
                    clean_entities = _clean_person_name(clean_entities)

                if label == "address":
                    clean_entities = _choose_first_address(clean_entities)

                if label in {"company", "university","degree"}:
                    clean_entities = _remove_substring_duplicates(clean_entities)

                # if label == "degree":
                #     clean_entities = _clean_degree_entities(clean_entities)

                # Remove duplicates ignoring case
                unique = {}
                for e in clean_entities:
                    key = e.lower()
                    if key not in unique:
                        unique[key] = e

                clean_entities = list(unique.values())

                row[label] = ", ".join(clean_entities) if clean_entities else None

            rows.append(row)

        except Exception as e:
            rows.append({"error": str(e)})

    return pd.DataFrame(rows)


