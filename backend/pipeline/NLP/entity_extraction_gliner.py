import re
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional, Any, cast, Tuple, Set
from pathlib import Path
from datetime import datetime

CURRENT_DATE = datetime.now()
from services.paths_models_and_data import get_gliner_model, SKILL_LABELS, COUNTRIES_AND_CAPITALS, COUNTRIES_NAMES, CAPITALS_NAMES, LANGUAGES, OUTPUTS_DIR

# ----------------------------------------------------------
# Gliner labels, regex, etc
# ----------------------------------------------------------

SIMPLE_JOB_TITLES = {
    # EN
    "teacher", "professor", "manager", "analyst", "engineer", "developer",
    "consultant", "coordinator", "supervisor", "assistant", "technician",
    "driver", "lifeguard", "operator", "pilot", "scientist", "private security",
    "security officer", "security guard", "deliveryman", "forester","biochemist",
    "food scientist", "geneticist","farmer", "biologist", "horticulturalist",
    "inspector","beekeeper","ecologist","conservationist", "veterinarian",
    "zoologist", "groomer", "animal control officer", "attendant", "pet walker",
    "animal sitter", "breeder", "pathologist","ophthalmologist", "reporter",
    "actuary","clerk", "artist", "therapist", "barber", "beautician", "designer",
    "stylist","cashier", "coordinator", "advisor", "writer", "editor", "illustrator",
    "actor","singer","producer","web developer","architect","animator", "painter",
    "tutor", "headteacher", "counsellor", "nurse", "doctor", "personal trainer",
    "pharmacist","dentist", "orthodontist", "dietitian", "housekeeper", "porter",
    "tour guide", "supervisor", "director",


    # PT
    "professor", "professora", "gestor", "gerente", "analista", "engenheiro",
    "engenheira", "desenvolvedor", "desenvolvedora", "consultor", "consultora",
    "coordenador", "coordenadora", "supervisor", "supervisora", "assistente",
    "técnico", "técnica", "motorista", "condutor", "nadador-salvador",
    "operador", "operadora", "piloto", "segurança",

    # ES
    "profesor", "profesora", "gerente", "analista", "ingeniero", "ingeniera",
    "desarrollador", "desarrolladora", "consultor", "consultora", "coordinador",
    "coordinadora", "supervisor", "supervisora", "asistente", "técnico",
    "técnica", "conductor", "socorrista", "operador", "operadora", "piloto",
    "seguridad",
}

SIMPLE_JOB_TITLE_REGEX = re.compile(
    rf"\b(?:{'|'.join(sorted(map(re.escape, SIMPLE_JOB_TITLES), key=len, reverse=True))})\b",
    re.IGNORECASE
)

COMPOUND_JOB_TITLE_PATTERNS = [
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}manager\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}coordinator\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}assistant\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}driver\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}technician\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}officer\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}guard\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}specialist\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}engineer\b",
    r"\bprivate security\b",
    r"\boperations coordinator\b",
    r"\bsite construction preparation manager\b",
    r"\bsite construction preparation assistant\b",
]

COMPOUND_JOB_TITLE_REGEXES = [
    re.compile(pattern, re.IGNORECASE) for pattern in COMPOUND_JOB_TITLE_PATTERNS
]

# Define label sets globally for reuse
DEGREE_LEVELS = [
    "high school",
    "associate",
    "bachelor",
    "master",
    "doctor",
    "doctorate",
    "phd",
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
    "job_title": ["job title", "professional role", "occupation", "position held in a company"],
    "company": ["company", "organization", "employer", "workplace company"],
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
        "person","first name", "certificate","last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background", "background","education", "experience", "hobbies", "kangaroo"},
    "skills": {
        "person","first name", "certificate", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies", "c1", "first certificate"},
    "degree": {"degree", "education", "diploma", "certificate", "executive education"},
    "company": {"company", "organization", "linkedin", "github", "linkdin"},
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

# ----------------------------------------------------------
# Gliner functions
# ----------------------------------------------------------

def _extract_compound_job_titles_regex(text: str) -> List[str]:
    """
    Extract common compound job titles such as 'operations coordinator'
    or 'private security'.
    """
    if not text:
        return []

    results = []
    seen = set()

    for regex in COMPOUND_JOB_TITLE_REGEXES:
        for match in regex.findall(text):
            title = " ".join(match.split()).strip()
            if not title:
                continue

            if not _is_valid_job_title(title):
                continue

            key = title.lower()
            if key not in seen:
                seen.add(key)
                results.append(title)

    return results

def _extract_simple_job_titles_from_lines(text: str) -> List[str]:
    """
    Look for simple job titles line-by-line, which is often more precise for CVs.
    """
    if not text:
        return []

    results = []
    seen = set()

    for line in text.splitlines():
        line_clean = " ".join(line.split()).strip()
        if not line_clean:
            continue

        for match in SIMPLE_JOB_TITLE_REGEX.findall(line_clean):
            title = " ".join(match.split()).strip()
            if not title:
                continue

            if not _is_valid_job_title(title):
                continue

            key = title.lower()
            if key not in seen:
                seen.add(key)
                results.append(title)

    return results

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
    - Language levels / certification levels
    - Generic proficiency phrases
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
        "academic background","academic","background","education","experience",
        "professional experience","hobbies","soft skills","hard skills","skills",
        "skill","fundamental","advanced user","basic user","independent user",
        "proficient user","user", "skill level", "level 1", "level 2", "level 3",
        "level 4", "level 5", "level 6", "level 7"
    }

    if t in GENERIC_BAD_SKILLS:
        return False

    # Language / certificate levels
    bad_skill_patterns = [
        r"^\s*[abc][12]\s*$",                  # C1, C2, B1, B2, A1, A2
        r"^\s*[abc][12]\s+and\s+[abc][12]\s*$",  # C1 and C2
        r"^\s*[abc][12]\s*/\s*[abc][12]\s*$",    # C1/C2
        r"\bfirst certificate\b",
        r"\badvanced user\b",
        r"\bbasic user\b",
        r"\bindependent user\b",
        r"\bproficient user\b",
        r"\bfundamental\b",
        r"^\s*level\s*\d+\s*$",     # Level 5
        r"^\s*skill\s+level\s*$",   # Skill Level
        r"\blevel\s*\d+\b",         # qualquer menção isolada de nível
        r"\bskill\s+level\b",       # expressão genérica
    ]

    if any(re.search(pattern, t, re.IGNORECASE) for pattern in bad_skill_patterns):
        return False

    if len(t) <= 2:
        return False

    return True

def _format_name_case(name: str) -> str:
    """
    Convert a name to title case token by token.
    Example:
        'GUILHERME RIBERT' -> 'Guilherme Ribert'
    """
    if not name:
        return name

    tokens = name.split()
    return " ".join(token.capitalize() for token in tokens)

def _clean_person_name(entities: List[str]) -> List[str]:
    """
    Clean and normalize person name entities.
    Prefer pairs of plausible person-name tokens and discard likely aliases/usernames.
    """
    cleaned_results = []

    suspicious_name_tokens = {
        "cv", "resume", "perfil", "profile", "user", "username",
        "linkedin", "github", "gitlab", "instagram", "facebook",
        "twitter", "x", "discord", "skype"
    }

    def looks_like_alias(token: str) -> bool:
        t = token.strip()

        if not t:
            return True

        lower = t.lower()

        if lower in suspicious_name_tokens:
            return True

        # very short weird tokens
        if len(t) <= 2:
            return True

        # tokens with no vowels are often handles/abbreviations
        if not re.search(r"[aeiouáéíóúàâêôãõäëïöü]", lower):
            return True

        # mixed odd casing like sNovaisg / mSd / xXfoo
        if not (t.isupper() or t.islower() or t.istitle()):
            return True

        return False

    for raw in entities:
        tokens = raw.replace(",", " ").split()
        tokens = [t for t in tokens if NAME_TOKEN_PATTERN.match(t)]

        if not tokens:
            continue

        # remove obvious aliases/usernames
        tokens = [t for t in tokens if not looks_like_alias(t)]

        if not tokens:
            continue

        # one-token case
        if len(tokens) == 1:
            if len(tokens[0]) >= 3:
                cleaned_results.append(_format_name_case(tokens[0]))
            continue

        # remove initials
        tokens = [t for t in tokens if len(t) > 1]

        if len(tokens) < 2:
            continue

        # normalize casing
        tokens = [t.capitalize() for t in tokens]

        # choose the first plausible adjacent pair
        chosen_pair = None
        for i in range(len(tokens) - 1):
            first = tokens[i]
            second = tokens[i + 1]

            if len(first) < 3 or len(second) < 3:
                continue

            chosen_pair = _format_name_case(f"{first} {second}")
            break

        if chosen_pair:
            cleaned_results.append(chosen_pair)

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
        "university", "college", "school", "academy",
        "manager", "engineer", "project",
        "certificate", "department",
        "father", "mother", "nationality",
        "address", "passport",
        "seeking", "position", "experience",
        "science", "data", "starter", "starters", "ldsa",
        "cambridge", "assessment", "english", "fct", "unl"}

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
            
            if len(first) == 1 or len(second) == 1:
                continue

            # Both must start uppercase
            if not (first[0].isupper() and second[0].isupper()):
                continue

            if first.lower() in forbidden_words:
                continue

            if second.lower() in forbidden_words:
                continue

            return _format_name_case(f"{first} {second}")

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

def _expand_address_smart(full_text: str, address_fragment: str, window: int = 80, capitals_names: set|None = CAPITALS_NAMES, countries_names: set|None=COUNTRIES_NAMES) -> str:
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
    Returns False for generic section titles, language/certificate labels,
    and strings with too many non-letter characters.
    """
    if not text:
        return False

    t = text.strip().lower()

    generic_bad_titles = {
        "professional experience","experience","work experience","professional summary",
        "summary","profile","about me","education","academic background","skills",
        "technical skills","languages","certifications","projects","contacts","contact",
    }

    if t in generic_bad_titles:
        return False

    # reject titles that are too generic if they contain these patterns
    bad_title_patterns = [
        r"\bexperience\b",
        r"\bsummary\b",
        r"\bprofile\b",
        r"\beducation\b",
        r"\bskills\b",
        r"\blanguages\b",
        r"\bcertifications?\b",
    ]

    if any(re.search(pattern, t, re.IGNORECASE) for pattern in bad_title_patterns):
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



# ----------------------------------------------------------
# Regex experience
# ----------------------------------------------------------
MONTH_PATTERN = (
    r"Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
    r"Sep|Sept|September|Oct|October|Nov|November|Dec|December|"
    r"Janeiro|Fev|Fevereiro|Março|Marco|Abr|Abril|Mai|Maio|Junho|Julho|Ago|Agosto|"
    r"Set|Setembro|Out|Outubro|Novembro|Dez|Dezembro|"
    r"Ene|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Septiembre|Octubre|Noviembre|Dic|Diciembre"
)

PRESENT_PATTERN = r"""Present|Current|Now|Today|Presente|Atual|Agora|Hoje|Actual|Ahora|Hoy"""

RANGE_CONNECTOR_PATTERN = r"""-|–|—|to|until|through|a|até|hasta"""

SINCE_PATTERN = r"""since|from|desde"""

MONTH_MAP = {
    # English
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,

    # Portuguese
    "fev": 2, "fevereiro": 2,
    "março": 3, "marco": 3,
    "abr": 4, "abril": 4,
    "mai": 5, "maio": 5,
    "junho": 6,
    "julho": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "setembro": 9,
    "out": 10, "outubro": 10,
    "novembro": 11,
    "dez": 12, "dezembro": 12,

    # Spanish
    "ene": 1, "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

PRESENT_TOKENS = {
    # EN
    "present", "current", "now", "today", "nowadays", "contemporary",

    # PT
    "presente", "atual", "agora", "hoje",

    # ES
    "actual", "ahora", "hoy", "presente",

    # FR
    "présent", "present", "actuel", "aujourd’hui", "aujourd'hui", "maintenant",

    # DE
    "heute", "aktuell", "derzeit", "jetzt",

    # SV
    "nuvarande", "idag", "nu",

    # IT
    "attuale", "oggi", "adesso", "presente",

    # NL
    "heden", "vandaag", "nu",

    # PL
    "obecnie", "teraz", "dziś", "dzisiaj",

    # TR
    "günümüz", "gunumuz", "bugün", "bugun", "şimdi", "simdi", "mevcut",

    # RU
    "сейчас", "сегодня", "настоящее время", "по настоящее время",

    # HI
    "वर्तमान", "अभी", "आज",

    # ZH
    "至今", "现在", "目前", "今",

    # JA
    "現在", "今", "現在まで",

    # KO
    "현재", "지금", "오늘",
}

DATE_RANGE_REGEX = re.compile(
    r"""
    (?:
        # optional prefixes
        since|from|year\s+|
        desde|de|
        depuis|seit|sedan|dal|da|sinds|od|beri|itibaren|с|से|自|从|から|부터
    )?
    \s*

    # ---------------- START DATE ----------------
    (?:
        (?:\d{1,2}[./-]){0,2}
        (?P<start_month>
            Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
            January|February|March|April|June|July|August|September|October|November|December|

            Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
            Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

            Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
            Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic|

            Janvier|Février|Fevrier|Mars|Avril|Mai|Juin|Juillet|Août|Aout|Septembre|Octobre|Novembre|Décembre|Decembre|
            Janv|Févr|Fevr|Avr|Juil|Déc|Dec|

            Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|
            Mär|Maerz|Okt|Dez|

            Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|
            Gen|Mag|Giu|Lug|Set|Ott|

            Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December|
            Mrt|Mei|Okt|

            Styczeń|Styczen|Luty|Marzec|Kwiecień|Kwiecien|Maj|Czerwiec|Lipiec|Sierpień|Sierpien|Wrzesień|Wrzesien|Październik|Pazdziernik|Listopad|Grudzień|Grudzien|
            Sty|Lut|Kwi|Cze|Lip|Sie|Wrz|Paź|Paz|Lis|Gru|

            Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik|
            Oca|Şub|Sub|Nis|Haz|Tem|Ağu|Agu|Eyl|Eki|Kas|Ara
        )?
        \s*
        (?P<start_year>\d{2,4})
        \s*(?:年|년)?
    )

    \s*
    (?:
        -|–|—|~|→|
        \bto\b|\buntil\b|\btill\b|\bthrough\b|
        \ba\b|\baté\b|\bhasta\b|\bà\b|\bau\b|\bbis\b|\btot\b|\bdo\b|\bkadar\b|\bпо\b|\bдо\b|\bतक\b|
        到|至|〜|～|까지
    )
    \s*

    # ---------------- END DATE ----------------
    (?:
        (?:\d{1,2}[./-]){0,2}
        (?P<end_month>
            Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
            January|February|March|April|June|July|August|September|October|November|December|

            Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
            Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

            Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
            Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic|

            Janvier|Février|Fevrier|Mars|Avril|Mai|Juin|Juillet|Août|Aout|Septembre|Octobre|Novembre|Décembre|Decembre|
            Janv|Févr|Fevr|Avr|Juil|Déc|Dec|

            Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|
            Mär|Maerz|Okt|Dez|

            Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|
            Gen|Mag|Giu|Lug|Set|Ott|

            Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December|
            Mrt|Mei|Okt|

            Styczeń|Styczen|Luty|Marzec|Kwiecień|Kwiecien|Maj|Czerwiec|Lipiec|Sierpień|Sierpien|Wrzesień|Wrzesien|Październik|Pazdziernik|Listopad|Grudzień|Grudzien|
            Sty|Lut|Kwi|Cze|Lip|Sie|Wrz|Paź|Paz|Lis|Gru|

            Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik|
            Oca|Şub|Sub|Nis|Haz|Tem|Ağu|Agu|Eyl|Eki|Kas|Ara
        )?
        \s*
        (?P<end_year>
            \d{2,4}|
            Present|Current|Today|Now|
            Presente|Atual|Hoje|Agora|
            Actual|Hoy|Ahora|
            Présent|Actuel|Aujourd['’]hui|Maintenant|
            Heute|Aktuell|Derzeit|Jetzt|
            Nuvarande|Idag|Nu|
            Attuale|Oggi|Adesso|
            Heden|Vandaag|
            Obecnie|Teraz|Dziś|Dzisiaj|
            Günümüz|Gunumuz|Bugün|Bugun|Şimdi|Simdi|Mevcut|
            Настоящее\s+время|Сейчас|Сегодня|По\s+настоящее\s+время|
            वर्तमान|अभी|आज|
            至今|现在|目前|今|
            現在|現在まで|
            현재|지금|오늘
        )
        \s*(?:年|년)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SINCE_REGEX = re.compile(
    r"""
    (?:
        since|from|
        desde|
        depuis|
        seit|
        sedan|
        dal|da|
        sinds|
        od|
        beri|itibaren|
        с|
        से|
        自|从|
        から|
        부터
    )
    \s*
    (?P<start_month>
        Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
        January|February|March|April|June|July|August|September|October|November|December|

        Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
        Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

        Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
        Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic
    )?
    \.?
    \s*
    (?P<start_year>\d{2,4})
    \s*(?:年|년)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

EXPLICIT_YEARS_REGEX = re.compile(
    r"""
    (?:
        # EN
        less\ than|under|more\ than|over|at\ least|minimum|min\.?|up\ to|approx\.?|around|about|
        # PT
        menos\ de|mais\ de|pelo\ menos|cerca\ de|aprox\.?|
        # ES
        menos\ de|más\ de|al\ menos|aprox\.?|cerca\ de|
        # FR
        moins\ de|plus\ de|au\ moins|environ|
        # DE
        weniger\ als|mehr\ als|mindestens|ca\.?|
        # SV
        mindre\ än|mer\ än|minst|cirka|
        # IT
        meno\ di|più\ di|almeno|circa|
        # NL
        minder\ dan|meer\ dan|minstens|ongeveer|
        # PL
        mniej\ niż|więcej\ niż|co\ najmniej|około|
        # TR
        en\ az|yaklaşık
    )?
    \s*

    (?P<years>\d{1,3})
    \+?
    \s*

    (?:
        # EN
        years?|yrs?|y|
        # PT
        anos?|ano|
        # ES
        años?|ano|
        # FR
        ans?|année?s?|
        # DE
        jahre?n?|jahr|
        # SV
        år|
        # IT
        anni|anno|
        # NL
        jaren?|jaar|
        # PL
        lat|rok|lata|
        # TR
        yıl|yil|
        # RU
        лет|года?|год|
        # HI
        साल|
        # ZH
        年|
        # JA
        年|
        # KO
        년
    )

    (?:
        \s*
        (?:
            # EN
            of\s+experience|experience|exp\.?|
            # PT
            de\s+experiência|experiência|exp\.?|
            # ES
            de\s+experiencia|experiencia|exp\.?|
            # FR
            d['’]expérience|expérience|exp\.?|
            # DE
            erfahrung|berufserfahrung|
            # SV
            erfarenhet|
            # IT
            esperienza|
            # NL
            ervaring|
            # PL
            doświadczenia|
            # TR
            deneyim|
            # RU
            опыта|опыт|
            # HI
            अनुभव|
            # ZH
            经验|
            # JA
            経験|
            # KO
            경험
        )
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

WORK_KEYWORDS = {
    "experience", "work", "employment", "career", "history",
    "job", "role", "company", "professional", "corporate",
    "intern", "internship",
    "engineer", "developer", "manager", "analyst",
    "consultant", "coordinator", "supervisor",
    "security", "guard", "officer",
    "driver", "deliveryman", "delivery",
    "lifeguard", "operations", "operator", "pilot",
    "construction", "site", "preparation",
    "technician", "assistant",
    "professor", "scientist", "teacher",
    "crew", "staff", "client", "clients",
    "speaker", "logistics", "deployment",
    "managed", "developed", "supervised", "led", "teaching",
}

EXPERIENCE_EDU_KEYWORDS = {
    "education", "academic", "student", "degree", "course",
    "bachelor", "master", "phd", "msc", "mba",
    "university", "academy", "school", "faculty",
    "graduated", "graduation", "thesis", "bootcamp",
}

NON_WORK_EXPERIENCE_KEYWORDS = {
    "project", "projects", "capstone", "training", "learning", "bootcamp",
    "skills", "tools", "technologies", "frameworks", "packages", "modeling",
    "volunteer", "sports", "hobbies","sailing",
    "music", "band", "bands", "runner", "running", "karate", "piano",
    "paraglider", "dancer", "scout",
    "non-corporate", "activities", "other activities",
}

WORK_SECTION_HEADERS = {
    "work experience",
    "professional experience",
    "corporate experience",
    "experiência profissional",
    "employment history",
    "career history",
    "professional background",
    "professional history",
    "professional career"
}

NON_WORK_SECTION_HEADERS = {
    "education",
    "education and training",
    "academic background",
    "projects",
    "project",
    "certifications",
    "skills",
    "competencies",
    "language skills",
    "languages",
    "volunteering",
    "other activities",
    "activities",
    "hobbies",
    "extra",
    "extra activities",
    "non-corporate experience",
    "music",
    "karaté",
    "karate",
    "runner",
}

# ----------------------------------------------------------
# Helpers to transform months and years
# ----------------------------------------------------------
def _parse_month(token: Optional[str], default: int) -> int:
    """
    Convert a month token to its numeric value.

    Examples:
        "Jan" -> 1
        "Maio" -> 5
        None -> default
        "unknown" -> default
    """
    if not token:
        return default

    normalized = token.strip().lower()
    return MONTH_MAP.get(normalized, default)


def _parse_year(token: Optional[str]) -> Optional[int]:
    """
    Convert a year token or present-token into a valid 4-digit year.
    Supports 2-digit and 4-digit years.
    """
    if not token:
        return None

    normalized = token.strip().lower()

    if normalized in PRESENT_TOKENS:
        return CURRENT_DATE.year

    if not re.fullmatch(r"\d{2,4}", normalized):
        return None

    year = int(normalized)

    if year < 100:
        year += 2000

    if 1950 <= year <= CURRENT_DATE.year + 1:
        return year

    return None

def _to_month_index(year: int, month: int) -> int:
    """
    Convert (year, month) to a sortable absolute month index.
    """
    return year * 12 + (month - 1)


def _merge_month_intervals(intervals: List[Tuple[int, int]]) -> int:
    """
    Merge overlapping month intervals and return total duration in months.

    Each interval is treated as [start, end), meaning end is exclusive.
    """
    if not intervals:
        return 0

    sorted_intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return sum(max(0, end - start) for start, end in merged)

# ----------------------------------------------------------
# Helpers for Experience
# ----------------------------------------------------------
def _get_nearby_lines(text: str, pos: int, lines_before: int = 3, lines_after: int = 1) -> str:
    """
    Return nearby lines around a character position.
    Useful to detect section headers near a date range.
    """
    if not text:
        return ""

    lines = text.splitlines()

    char_count = 0
    line_idx = 0

    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    start = max(0, line_idx - lines_before)
    end = min(len(lines), line_idx + lines_after + 1)

    return "\n".join(lines[start:end]).lower()

def _classify_section_near_date(text: str, pos: int) -> str:
    """
    Classify a date by the nearest relevant section header above it.
    Returns:
    - 'work'
    - 'non_work'
    - ''
    """
    if not text:
        return ""

    lines = text.splitlines()

    char_count = 0
    line_idx = 0
    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    # look upwards for the nearest meaningful header
    for i in range(line_idx, -1, -1):
        line = " ".join(lines[i].split()).strip().lower()
        if not line:
            continue

        for header in NON_WORK_SECTION_HEADERS:
            if re.fullmatch(rf"{re.escape(header)}", line):
                return "non_work"

        for header in WORK_SECTION_HEADERS:
            if re.fullmatch(rf"{re.escape(header)}", line):
                return "work"

    return ""

def clean_text_keep_lines(text: str) -> str:
    """
    Normalize text while preserving line structure.

    - keeps line breaks
    - removes common bullet/tab noise
    - collapses repeated spaces
    - collapses excessive blank lines
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\uf0b7\u2022*]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    """
    Return the start and end indexes of the line containing `pos`.
    """
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)

    if line_end == -1:
        line_end = len(text)

    return line_start, line_end

def _match_has_experience_word(match_text: str) -> bool:
    if not match_text:
        return False

    return bool(re.search(
        r"""
        \b(
            experience|experiences|exp\.?|
            experiência|experiencias?|experiencia|
            expérience|
            erfahrung|berufserfahrung|
            erfarenhet|
            esperienza|
            ervaring|
            doświadczenia|
            deneyim|
            опыт|опыта|
            अनुभव|
            经验|経験|경험
        )\b
        """,
        match_text,
        re.IGNORECASE | re.VERBOSE,
    ))

def _get_local_context(
    text: str,
    pos: int,
    left_words: int = 25,
    right_words: int = 25,
) -> str:
    """
    Extract a word-based local context window around a character position.
    """
    words = re.split(r"\s+", text)

    char_count = 0
    idx = len(words) - 1 if words else 0

    for i, word in enumerate(words):
        char_count += len(word) + 1
        if char_count >= pos:
            idx = i
            break

    start = max(0, idx - left_words)
    end = min(len(words), idx + right_words)

    return " ".join(words[start:end]).lower()

def _tokenize_context(text: str) -> Set[str]:
    """
    Convert context text into a normalized token set.
    """
    return {
        re.sub(r"[^\w]", "", token.lower())
        for token in re.split(r"\s+", text)
        if token and re.sub(r"[^\w]", "", token.lower())
    }

def _is_valid_explicit_experience_match(
    text: str,
    pos: int,
    years: int,
    academic_keywords: Set[str],
    non_work_keywords: Set[str],
) -> bool:
    """
    Decide if an explicit expression like '5 years experience'
    is likely to refer to professional experience.
    """
    if years <= 0 or years > 50:
        return False

    local_ctx = _get_local_context(text, pos, left_words=30, right_words=30)

    hard_negative_patterns = [
        r"\bthesis experience\b",
        r"\bproject experience\b",
        r"\bprojects experience\b",
        r"\bnon-corporate\b",
        r"\bother activities\b",
        r"\bflying hours\b",
        r"\bkm per week\b",
        r"\bvolunteer\b",
        r"\bstudent\b",
        r"\bcourse\b",
        r"\bacademy\b",
        r"\bbootcamp\b",
    ]

    if any(re.search(pattern, local_ctx, re.IGNORECASE) for pattern in hard_negative_patterns):
        return False

    token_set = _tokenize_context(local_ctx)

    if token_set & academic_keywords:
        return False

    if token_set & non_work_keywords:
        return False

    work_signal_patterns = [
        r"\bwork(?:ing)? experience\b",
        r"\bprofessional experience\b",
        r"\bindustry experience\b",
        r"\bcorporate experience\b",
        r"\bcurrently working\b",
        r"\bjob\b",
        r"\brole\b",
        r"\bconsultant\b",
        r"\bengineer\b",
        r"\bscientist\b",
        r"\bdeveloper\b",
        r"\bmanager\b",
        r"\banalyst\b",
        r"\bprofessor\b",
        r"\bteacher\b",
        r"\bintern(?:ship)?\b",
        r"\bsupervisor\b",
        r"\bcoordinator\b",
        r"\bdriver\b",
        r"\bsecurity\b",
        r"\bguard\b",
        r"\bofficer\b",
        r"\blifeguard\b",
        r"\boperations?\b",
        r"\bconstruction\b",
        r"\bsite\b",
        r"\btechnician\b",
        r"\bassistant\b",
        r"\bpilot\b",
        r"\boperator\b",
        r"\bdeliveryman\b",
        r"\bdelivery\b",
    ]

    return any(re.search(pattern, local_ctx, re.IGNORECASE) for pattern in work_signal_patterns)

def _build_academic_context_keywords(
    base_education_keywords: Set[str],
    degrees: List[str] | None = None,
    universities: List[str] | None = None,
) -> Set[str]:
    keywords = set(base_education_keywords)

    for item in (degrees or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for item in (universities or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for key in DEGREE_NORMALIZATION_MAP.keys():
        for token in re.findall(r"\w+", key.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for level in DEGREE_LEVELS:
        keywords.add(level.lower())

    return keywords

def _context_has_keywords(
    text: str,
    pos: int,
    direction: str,
    window_words: int,
    academic_keywords: Set[str],
    work_keywords: Set[str],
) -> str:
    words = re.split(r"\s+", text)

    char_count = 0
    idx = len(words) - 1 if words else 0

    for i, word in enumerate(words):
        char_count += len(word) + 1
        if char_count >= pos:
            idx = i
            break

    if direction == "right":
        segment = words[idx: idx + window_words]
    elif direction == "left":
        segment = words[max(0, idx - window_words): idx]
    else:
        return ""

    segment_set = {
        re.sub(r"[^\w]", "", word.lower())
        for word in segment
        if word and re.sub(r"[^\w]", "", word.lower())
    }

    if segment_set & academic_keywords:
        return "edu"

    if segment_set & work_keywords:
        return "work"

    return ""

def _build_job_title_keywords(job_titles: List[str] | None = None) -> Set[str]:
    keywords: Set[str] = set()

    for item in (job_titles or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 3:
                keywords.add(token)

    return keywords

def _nearby_lines_have_job_title(
    text: str,
    pos: int,
    job_titles: List[str] | None = None,
    window_lines: int = 2
) -> bool:
    """
    Check whether nearby lines around a date contain a likely job title.
    """
    if not text:
        return False

    lines = text.splitlines()

    char_count = 0
    line_idx = 0

    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    start = max(0, line_idx - window_lines)
    end = min(len(lines), line_idx + window_lines + 1)

    nearby_lines = []
    for line in lines[start:end]:
        line_clean = " ".join(line.split()).strip()
        if line_clean:
            nearby_lines.append(line_clean)

    # 1) direct match against extracted job titles
    extracted_titles = [jt.strip().lower() for jt in (job_titles or []) if jt and jt.strip()]
    for line_clean in nearby_lines:
        line_lower = line_clean.lower()

        if line_lower in extracted_titles:
            return True

        if any(jt in line_lower or line_lower in jt for jt in extracted_titles):
            return True

    # 2) regex fallback
    for line_clean in nearby_lines:
        if not _is_valid_job_title(line_clean):
            continue

        if SIMPLE_JOB_TITLE_REGEX.search(line_clean):
            return True

        for rx in COMPOUND_JOB_TITLE_REGEXES:
            if rx.search(line_clean):
                return True

    return False

# ----------------------------------------------------------
# Experience functions
# ----------------------------------------------------------
def extract_experience_years(
    raw_text: str,
    degrees: List[str] | None = None,
    universities: List[str] | None = None,
    job_titles: List[str] | None = None,
) -> Optional[int]:
    """
    Estimate the total number of professional experience years from CV text.

    The estimation combines multiple complementary strategies:

    1) Explicit mentions of experience duration
       Examples:
           "5 years", "7+ years of experience", "over 3 yrs"

    2) Closed date ranges describing employment periods
       Examples:
           "2019–2024", "May 2017 - April 2019", "2020 to Present"

    3) Open date ranges using "since"
       Examples:
           "since 2020", "working since 2018"

    4) Semantic context filtering
       - Excludes education-related periods
       - Keeps work/professional periods
       - Uses:
           • short right-side context
           • long left-side context

    5) Interval merging
       - Prevents double counting overlapping jobs

    Returns
    -------
    Optional[int]
        Total estimated years of professional experience.
        Returns None if no reliable estimation is possible.
    """

    if not raw_text:
        return None

    text = clean_text_keep_lines(raw_text)

    academic_keywords = _build_academic_context_keywords(
        base_education_keywords=EXPERIENCE_EDU_KEYWORDS,
        degrees=degrees,
        universities=universities,
    )

    job_title_keywords = _build_job_title_keywords(job_titles)
    dynamic_work_keywords = set(WORK_KEYWORDS) | job_title_keywords

    # ------------------------------------------------------
    # 1) Explicit mentions (e.g. "5 years of experience")
    # ------------------------------------------------------
    explicit_years: List[int] = []
    line_best_matches: dict[tuple[int, int], re.Match[str]] = {}

    for match in EXPLICIT_YEARS_REGEX.finditer(text):
        pos = match.start()
        line_span = _line_bounds(text, pos)

        current_best = line_best_matches.get(line_span)
        if current_best is None:
            line_best_matches[line_span] = match
            continue

        current_has_exp = _match_has_experience_word(current_best.group(0))
        new_has_exp = _match_has_experience_word(match.group(0))

        # Prefer the one that explicitly includes "experience"
        if new_has_exp and not current_has_exp:
            line_best_matches[line_span] = match

    for match in line_best_matches.values():
        years = int(match.group("years"))
        pos = match.start()

        if _is_valid_explicit_experience_match(
            text=text,
            pos=pos,
            years=years,
            academic_keywords=academic_keywords,
            non_work_keywords=NON_WORK_EXPERIENCE_KEYWORDS,
        ):
            explicit_years.append(years)

    # ------------------------------------------------------
    # 2) Closed date ranges (start -> end)
    # ------------------------------------------------------
    intervals: List[Tuple[int, int]] = []

    for match in DATE_RANGE_REGEX.finditer(text):
        start_year = _parse_year(match.group("start_year"))
        end_year_raw = match.group("end_year")
        if not end_year_raw:
            continue

        end_year = _parse_year(end_year_raw)

        if start_year is None or end_year is None:
            continue

        start_month = _parse_month(match.group("start_month"), 1)

        is_present = end_year_raw.strip().lower() in PRESENT_TOKENS
        end_month = _parse_month(
            None if is_present else match.group("end_month"),
            CURRENT_DATE.month if is_present else 12,
        )

        start_idx = _to_month_index(start_year, start_month)
        end_idx = _to_month_index(end_year, end_month)

        # Invalid reversed range
        if end_idx < start_idx:
            continue

        pos = match.start()

        section_cls = _classify_section_near_date(text, pos)

        if section_cls == "non_work":
            continue

        if _nearby_lines_have_job_title(text, pos, job_titles=job_titles, window_lines=2):
            intervals.append((start_idx, end_idx + 1))
            continue

        local_ctx = _get_local_context(text, pos, left_words=40, right_words=20)
        token_set = _tokenize_context(local_ctx)

        if section_cls == "work":
            intervals.append((start_idx, end_idx + 1))
            continue

        if token_set & NON_WORK_EXPERIENCE_KEYWORDS:
            continue

        right_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="right",
            window_words=12,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "edu":
            continue

        if right_ctx == "work":
            intervals.append((start_idx, end_idx + 1))
            continue

        left_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="left",
            window_words=100,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if left_ctx == "work":
            intervals.append((start_idx, end_idx + 1))

    # ------------------------------------------------------
    # 3) Open intervals like "since 2020"
    # ------------------------------------------------------
    for match in SINCE_REGEX.finditer(text):
        start_year = _parse_year(match.group("start_year"))
        if start_year is None:
            continue

        start_month = _parse_month(match.group("start_month"), 1)
        start_idx = _to_month_index(start_year, start_month)
        end_idx = _to_month_index(CURRENT_DATE.year, CURRENT_DATE.month)

        if end_idx < start_idx:
            continue

        pos = match.start()

        section_cls = _classify_section_near_date(text, pos)

        if section_cls == "non_work":
            continue

        if _nearby_lines_have_job_title(text, pos, job_titles=job_titles, window_lines=2):
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)
            continue

        local_ctx = _get_local_context(text, pos, left_words=40, right_words=20)
        token_set = _tokenize_context(local_ctx)

        if section_cls == "work":
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)
            continue

        if token_set & NON_WORK_EXPERIENCE_KEYWORDS:
            continue

        right_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="right",
            window_words=12,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "edu":
            continue

        left_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="left",
            window_words=100,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "work" or left_ctx == "work":
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)

    # ------------------------------------------------------
    # 4) Merge intervals and compute total months
    # ------------------------------------------------------
    total_months = _merge_month_intervals(intervals)

    if total_months < 0:
        return None

    # Reject unrealistic values
    if total_months > 50 * 12:
        return None

    # ------------------------------------------------------
    # 5) Fallback to explicit mentions if no interval found
    # ------------------------------------------------------
    if total_months == 0:
        if explicit_years:
            best_explicit = max(explicit_years)
            if best_explicit <= 15:
                return best_explicit
        return None

    # Convert merged months to years
    return int(total_months / 12 + 0.5)


# ----------------------------------------------------------
# MAIN FUNCTIONS
# ----------------------------------------------------------
def extract_entities_from_text(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.15,
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
        # Get broader predictions with a lower threshold, then filter manually
        # ---------------------------------------
        raw_threshold = 0.10

        all_entities = model.predict_entities(
            text,
            ALL_LABELS,
            threshold=raw_threshold
        )  # type: ignore[attr-defined]
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
        entities = []

        for ent in all_entities:

            label = ent.get("label")
            score = ent.get("score", 0.0)

            # person name: threshold mais permissivo
            if label in ENTITY_GROUPS["person_name"]:
                if score >= 0.1:
                    entities.append(ent)

            else:
                if score >= threshold:
                    entities.append(ent)
                    
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
        
    # Fallback regex for job titles missed by GLiNER
    existing_job_titles = {t.lower() for t in entities_by_label["job_title"]}

    regex_job_titles = _extract_simple_job_titles_from_lines(text)
    regex_job_titles += _extract_compound_job_titles_regex(text)

    for title in regex_job_titles:
        key = title.lower()
        if key not in existing_job_titles:
            entities_by_label["job_title"].append(title)
            existing_job_titles.add(key)

    return entities_by_label

def process_texts_gliner(
    texts: Union[List[str], List[Dict[str, str]], Dict[str, Any],],
    model: Optional[GLiNER] = None,
    gliner_threshold: float = 0.15,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR
) -> pd.DataFrame:

    if model is None:
        model = get_gliner_model()

    if isinstance(texts, list):
        text_items = list(enumerate(texts))
    elif isinstance(texts, dict):
        if any(k in texts for k in ("gliner_text", "raw_text", "text")):
            text_items = [(0, texts)]
        else:
            text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")

    rows = []

    for _, item in text_items:
        try:
            if isinstance(item, dict):
                gliner_text = item.get("gliner_text", "") or item.get("text", "")
                raw_text = item.get("raw_text", "") or gliner_text
            else:
                gliner_text = item
                raw_text = item

            entities_dict = extract_entities_from_text(
                gliner_text,
                model=model,
                threshold=gliner_threshold,
                debug_save_json=debug_save_json,
                debug_output_path=debug_output_path
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
                    heuristic_name = extract_top_name_candidate(raw_text)

                    if not clean_entities:
                        if heuristic_name:
                            clean_entities = [heuristic_name]
                    else:
                        if heuristic_name:
                            first_found = clean_entities[0]

                            # só substitui se GLiNER encontrou um único nome
                            # e esse nome aparecer dentro do nome heurístico
                            if len(first_found.split()) == 1 and len(heuristic_name.split()) >= 2:
                                first_token = first_found.lower()
                                heuristic_tokens = [t.lower() for t in heuristic_name.split()]

                                if first_token in heuristic_tokens:
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
            
            row["experience_years"] = extract_experience_years(
                raw_text=raw_text,
                degrees=row.get("degree") or [],
                universities=row.get("university") or [],
                job_titles=row.get("job_title") or [],
            ) or 0

            # Convert lists to string
            for key in row:
                if isinstance(row[key], list):
                    row[key] = ", ".join(row[key]) if row[key] else None


            rows.append(row)

        except Exception as e:
            rows.append({"error": str(e)})

    return pd.DataFrame(rows)

