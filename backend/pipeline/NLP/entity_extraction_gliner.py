import re
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional, Any, cast
from pathlib import Path

from services.paths_models_and_data import get_gliner_model, SKILL_LABELS, COUNTRIES_AND_CAPITALS, COUNTRIES_NAMES, CAPITALS_NAMES, LANGUAGES, OUTPUTS_DIR


# Define label sets globally for reuse
DEGREE_LEVELS = [
    "highschool",
    "associate",
    "bachelor",
    "master",
    "doctor",
]

DEGREE_NORMALIZATION_MAP = {
    # --------------------
    # Secondary / High School
    # --------------------
    "highschool": "High School Diploma",
    "high school": "High School Diploma",
    "secondary school": "High School Diploma",
    "hs diploma": "High School Diploma",
    "diploma do ensino secundário": "High School Diploma",
    "ensino secundário": "High School Diploma",
    "12º year": "High School Diploma",
    "12º ano": "High School Diploma",
    "12o ano": "High School Diploma",

    # --------------------
    # Associate / Short-cycle
    # --------------------
    "associate": "Associate Degree",
    "a.a": "Associate of Arts",
    "aa": "Associate of Arts",
    "a.s": "Associate of Science",
    "as": "Associate of Science",
    "cets": "Short-cycle Higher Education (CTeSP)",
    "ctesp": "Short-cycle Higher Education (CTeSP)",
    "curso técnico superior profissional": "Short-cycle Higher Education (CTeSP)",

    # --------------------
    # Bachelor
    # --------------------
    "bachelor": "Bachelor’s Degree",
    "bachelors": "Bachelor’s Degree",
    "undergraduate": "Bachelor’s Degree",
    "graduação": "Bachelor’s Degree",
    "licenciatura": "Bachelor’s Degree",
    "licenciado": "Bachelor’s Degree",

    # Bachelor abbreviations
    "b.sc": "Bachelor of Science",
    "bsc": "Bachelor of Science",
    "b.s": "Bachelor of Science",
    "bs": "Bachelor of Science",
    "b.eng": "Bachelor of Engineering",
    "beng": "Bachelor of Engineering",
    "b.e": "Bachelor of Engineering",
    "be": "Bachelor of Engineering",
    "b.tech": "Bachelor of Technology",
    "btech": "Bachelor of Technology",
    "b.a": "Bachelor of Arts",
    "ba": "Bachelor of Arts",
    "bba": "Bachelor of Business Administration",
    "llb": "Bachelor of Laws",

    # --------------------
    # Master
    # --------------------
    "master": "Master’s Degree",
    "masters": "Master’s Degree",
    "postgraduate": "Master’s Degree",
    "pós-graduação": "Postgraduate Qualification",
    "pos-graduação": "Postgraduate Qualification",
    "pos graduacao": "Postgraduate Qualification",
    "pós graduação": "Postgraduate Qualification",
    "mestrado": "Master’s Degree",
    "mestre": "Master’s Degree",

    # Master abbreviations
    "m.sc": "Master of Science",
    "msc": "Master of Science",
    "m.s": "Master of Science",
    "ms": "Master of Science",
    "m.eng": "Master of Engineering",
    "meng": "Master of Engineering",
    "m.e": "Master of Engineering",
    "me": "Master of Engineering",
    "m.tech": "Master of Technology",
    "mtech": "Master of Technology",
    "m.a": "Master of Arts",
    "ma": "Master of Arts",
    "mba": "Master of Business Administration",
    "llm": "Master of Laws",

    # Integrated Master (PT)
    "mestrado integrado": "Integrated Master’s Degree",

    # --------------------
    # Doctorate
    # --------------------
    "doctorate": "Doctorate",
    "doctoral": "Doctorate",
    "doutoramento": "Doctorate",
    "doutorado": "Doctorate",

    # Doctorate abbreviations
    "ph.d": "Doctor of Philosophy",
    "phd": "Doctor of Philosophy",
    "d.phil": "Doctor of Philosophy",
    "dphil": "Doctor of Philosophy",
    "md": "Doctor of Medicine",
    "m.d": "Doctor of Medicine",
    "jd": "Juris Doctor",
    "j.d": "Juris Doctor",
}

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
    "davinci resolve"}

ENTITY_GROUPS = {
    "skills": SKILL_LABELS,
    "person_name": ["person name (individual human name)"],
    "job_title": ["job title", "professional role", "occupation"],
    "company": ["company", "organization", "employer"],
    "university": ["university", "higher education institution", "college"],
    "degree": ["degree"],
    "address": ["full address", "residential address", "home address"],}

ALL_LABELS = list(dict.fromkeys(
    label
    for labels in ENTITY_GROUPS.values()
    for label in labels
))

ACADEMIC_KEYWORDS = {
    "mestrado",
    "licenciatura",
    "doutoramento",
    "doutorado",
    "bachelor",
    "master",
    "phd",
    "degree",
    "engenharia",
    "engineering",
    "industrial",}

INVALID_ENTITIES = {
    "person_name": {
        "person", "certificate","last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies"},
    "job_title": {
        "person","first name", "certificate","last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background", "background","education", "experience", "hobbies"},
    "skills": {
        "person","first name", "certificate", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies", "C1", "First Certificate"},
    "degree": {"degree", "education", "diploma", "certificate", "executive education"},
    "company": {"company", "organization"},
    "address": {"address", "location"}}


GENERIC_INVALID = {
    "person",
    "certificate",
    "education",
    "experience",
    "skills",
    "skill",
    "hobbies",
}

POSTAL_PATTERN = re.compile(r"\b\d{4,6}[- ]?\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\b(nº|no\.?|number)?\s?\d+\b", re.IGNORECASE)

NAME_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")

# Normalize language list once
LANGUAGES_SET = {lang.lower().strip() for lang in LANGUAGES}

def _ensure_default_degree(university: List[str], degree: List[str]) -> List[str]:
    """
    If university exists and no known degree level is found,
    add default Bachelor's Degree.
    """

    if not university:
        return degree

    if not degree:
        return ["Bachelor’s Degree"]

    degree_text = " ".join(degree).lower()

    if not any(level in degree_text for level in DEGREE_LEVELS):
        return degree + ["Bachelor’s Degree"]

    return degree

def _cleanup_degree_phrase(text: str) -> str:
    """
    Remove duplicated or noisy 'Degree' tokens after normalization.

    Examples:
        "Bachelor’s Degree Degree" -> "Bachelor’s Degree"
        "Master’s Degree’s Degree" -> "Master’s Degree"
        "Bachelor Degree" -> "Bachelor’s Degree"  (optional normalization)
    """
    if not text:
        return text

    t = text

    # Normalize common noisy variants (optional but helpful)
    # e.g. "Bachelor Degree" -> "Bachelor’s Degree"
    t = re.sub(r"(?i)\b(bachelor)\s+degree\b", "Bachelor’s Degree", t)
    t = re.sub(r"(?i)\b(master)\s+degree\b", "Master’s Degree", t)
    t = re.sub(r"(?i)\b(doctor)\s+degree\b", "Doctorate", t)

    # Remove repeated "Degree" tokens (case-insensitive)
    # e.g. "Degree Degree" / "Degree’s Degree" / "Degree's Degree"
    t = re.sub(r"(?i)\bdegree(?:['’]s)?\b(?:\s+\bdegree(?:['’]s)?\b)+", "Degree", t)

    # Fix the frequent bug: "Bachelor’s Degree Degree" -> "Bachelor’s Degree"
    # Keep the first and drop the duplicated trailing Degree
    t = re.sub(r"(?i)\b(bachelor(?:['’]s)?\s+degree)\s+degree\b", r"\1", t)
    t = re.sub(r"(?i)\b(master(?:['’]s)?\s+degree)\s+degree\b", r"\1", t)

    # Cleanup extra spaces
    t = re.sub(r"\s+", " ", t).strip()

    return t

def _normalize_degree_entries(degrees: List[str]) -> List[str]:
    """
    Normalize degree abbreviations/translations while preserving the remaining text.

    Examples:
        "BSc in Computer Science" -> "Bachelor of Science in Computer Science"
        "Licenciatura em Engenharia Informática" -> "Bachelor’s Degree em Engenharia Informática"
        "M.Sc Data Science" -> "Master of Science Data Science"
    """
    if not degrees:
        return []

    # Prefer longer keys first (e.g., "mestrado integrado" before "mestrado")
    keys_sorted = sorted(DEGREE_NORMALIZATION_MAP.keys(), key=len, reverse=True)

    normalized: List[str] = []

    for degree in degrees:
        original = degree.strip()
        if not original:
            continue

        text = original
        lower = text.lower()

        replaced_any = False

        for key in keys_sorted:
            full_form = DEGREE_NORMALIZATION_MAP[key]

            # Replace occurrences as whole tokens (case-insensitive)
            pattern = rf"(?i)\b{re.escape(key)}\b"

            if re.search(pattern, lower):
                text = re.sub(pattern, full_form, text)
                replaced_any = True
                lower = text.lower()  # keep in sync after replacement

        # Cleanup spacing around punctuation
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:])", r"\1", text)  # remove spaces before punctuation

        # Remove duplicated Degree tokens after normalization
        text = _cleanup_degree_phrase(text)

        normalized.append(text if (replaced_any or text) else original)

    # Deduplicate case-insensitively while preserving order
    seen = set()
    out: List[str] = []
    for item in normalized:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            out.append(item)

    return out

def _is_valid_skill(text: str) -> bool:
    """
    Validate if a skill candidate is acceptable.

    Removes:
    - Languages
    - Generic placeholders
    - Academic degrees wrongly classified as skills
    """

    if not text:
        return False

    t = text.lower().strip()

    # Remove academic degree phrases
    for keyword in ACADEMIC_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", t):
            return False

    # Remove languages
    if t in LANGUAGES_SET:
        return False

    for lang in LANGUAGES_SET:
        if re.search(rf"\b{re.escape(lang)}\b", t):
            return False

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

def extract_top_name_candidate(text: str, window: int = 300) -> Optional[str]:
    """
    Strong fallback name extractor.
    Looks for a two-token Proper Case pattern inside the top section.
    """

    top = text[:window]

    # Normalize dots
    top = re.sub(r"\.", " ", top)

    lines = [l.strip() for l in top.split("\n") if l.strip()]

    forbidden_words = {
        "area", "interests", "objective", "profile", "summary",
        "university", "college", "school",
        "manager", "engineer", "project",
        "certificate", "department",
        "father", "mother", "nationality",
        "address", "passport",
        "seeking", "position", "experience"
    }

    noise_patterns = [
        r"@",
        r"\+?\d",
        r"\bmobile\b",
        r"\bemail\b",
        r"\bskype\b",
        r"\bphone\b",
    ]

    for line in lines[:8]:

        if any(re.search(p, line, re.IGNORECASE) for p in noise_patterns):
            continue

        tokens = line.split()

        # Only alphabetic tokens
        tokens = [t for t in tokens if NAME_TOKEN_PATTERN.match(t)]

        if len(tokens) < 2:
            continue

        # Slide window of 2 tokens inside the line
        for i in range(len(tokens) - 1):
            first = tokens[i]
            second = tokens[i + 1]

            # Both must start uppercase
            if not (first[0].isupper() and second[0].isupper()):
                continue

            if first.lower() in forbidden_words:
                continue

            if second.lower() in forbidden_words:
                continue

            return f"{first} {second}"

    return None

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

def _is_valid_job_title(text: str, threshold: float = 0.4) -> bool:
    """
    Returns False if more than `threshold` (default 40%)
    of the characters are digits or symbols.
    """
    if not text:
        return False

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False

    total_chars = len(chars)

    non_letter_chars = sum(
        1 for c in chars
        if not c.isalpha()
    )

    ratio = non_letter_chars / total_chars

    return ratio <= threshold

def extract_entities_from_text(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.3,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR) -> Dict[str, List[str]]:

    result_labels = [
                "person_name",
                "job_title",
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
        # ---------------------------------------
        # Always get full predictions (threshold=0)
        # ---------------------------------------
        if debug_save_json:
            all_entities = model.predict_entities(text, ALL_LABELS, threshold=0.1)# type: ignore[attr-defined]
        else:
            all_entities = model.predict_entities(text, ALL_LABELS, threshold=threshold)# type: ignore[attr-defined]
        # ---------------------------------------
        # Optional debug JSON save
        # ---------------------------------------
        if debug_save_json:
            import json

            debug_data = []

            for ent in all_entities:
                debug_data.append({
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": float(ent.get("score", 0.0))
                })

            if debug_output_path:
                output_path = Path(debug_output_path)

                if output_path.is_dir() or not output_path.suffix:
                    output_file = output_path / "gliner_thresholds.json"
                else:
                    output_file = output_path
            else:
                output_file = Path(OUTPUTS_DIR) / "gliner_thresholds.json"

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False)

        # ---------------------------------------
        # Apply threshold manually
        # ---------------------------------------
        entities = [
            ent for ent in all_entities
            if ent.get("score", 0.0) >= threshold
        ]
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
        group_invalid = set()

        for group_name, group_labels in ENTITY_GROUPS.items():
            if label in group_labels:
                group_invalid = INVALID_ENTITIES.get(group_name, set())
                break

        if text_norm in group_invalid:
            continue
        if text_norm in GENERIC_INVALID:
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
    gliner_threshold: float = 0.3,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR) -> pd.DataFrame:

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
                threshold=gliner_threshold,
                debug_save_json=debug_save_json,
                debug_output_path= debug_output_path
            )

            row: Dict[str, Any] = {}

            for label in [
                "person_name",
                "job_title",
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

                if label == "person_name":
                    clean_entities = _clean_person_name(clean_entities)

                    if not clean_entities:
                        heuristic_name = extract_top_name_candidate(text)
                        if heuristic_name:
                            clean_entities = [heuristic_name]

                if label == "address":
                    clean_entities = _choose_first_address(clean_entities)

                if label == "job_title":
                    clean_entities = [
                        e for e in clean_entities
                        if _is_valid_job_title(e)
                    ]

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

                row[label] = clean_entities
            
            # Normalize degrees
            row["degree"] = _normalize_degree_entries(row.get("degree") or [])

            # Use the default degree
            row["degree"] = _ensure_default_degree(
                row.get("university") or [],
                row.get("degree") or [])

            # Convert lists to string
            for key in row:
                if isinstance(row[key], list):
                    row[key] = ", ".join(row[key]) if row[key] else None


            rows.append(row)

        except Exception as e:
            rows.append({"error": str(e)})

    return pd.DataFrame(rows)


