import re
import pandas as pd
from typing import Dict, List, Optional, Union, TypedDict, Tuple, Mapping, Sequence, Any
from re import Pattern

from datetime import datetime
CURRENT_YEAR = datetime.now().year

import pycountry
from services.paths_models_and_data import LANGUAGES, COUNTRIES_AND_CAPITALS 

import phonenumbers
from phonenumbers import NumberParseException

# ----------------------------------------------------------
# Regex language and language level vocabularies
# ----------------------------------------------------------

LANG_PATTERN = re.compile(r"\b(" + "|".join(sorted((re.escape(l) for l in LANGUAGES),key=len,reverse=True)) + r")\b",re.IGNORECASE)

LANG_LEVEL_MAP = {
    "Beginner": {
        # CEFR
        "a1", "a2",

        # Common CV terms
        "beginner", "basic", "basics", "elementary",
        "low proficiency", "low level", "starter",

        # Academic / school wording
        "introductory level", "introductory",
        "foundational level", "foundation level",
        "initial level",

        # ILR / ACTFL (mapped conservatively)
        "elementary proficiency",
        "limited proficiency",
        "limited working proficiency",

        # Self-assessment
        "basic understanding",
        "basic knowledge",
        "some knowledge"
    },

    "Intermediate": {
        # CEFR
        "b1", "b2",

        # Common CV terms
        "intermediate", "medium proficiency", "medium level",
        "conversational",
        "working knowledge",

        # Academic / school wording
        "intermediate level",
        "upper intermediate",
        "lower intermediate",

        # ILR / ACTFL
        "professional working proficiency",
        "working proficiency",

        # Self-assessment
        "good knowledge",
        "comfortable using",
        "functional proficiency"
    },

    "Advanced": {
        # CEFR
        "c1", "c2",

        # Common CV terms
        "advanced", "high proficiency", "high level",
        "fluent", "fluent speaker",
        "proficient",
        "full professional proficiency",

        # Academic / school wording
        "advanced level",
        "upper advanced",

        # ILR / ACTFL
        "full proficiency",

        # Near-native wording
        "near native", "near-native",
        "almost native"
    },

    "Native": {
        # Direct
        "native", "native speaker",

        # Academic
        "mother tongue", "mother-tongue",
        "first language", "l1",

        # CV wording
        "bilingual",
        "native proficiency",

        # ACTFL / self-identification
        "heritage speaker"
    },
}

LANG_LOOKUP = {l.lower(): l for l in LANGUAGES}

LEVEL_LOOKUP = {term: level
    for level, terms in LANG_LEVEL_MAP.items()
    for term in terms}

LEVEL_PRIORITY = {"Beginner": 1,"Intermediate": 2,"Advanced": 3,"Native": 4,}

LEVEL_RANK = LEVEL_PRIORITY

CEFR_RANK = {"A0": 0,"A1": 1,"A2": 2,"B1": 3,"B2": 4,"C1": 5,"C2": 6,}

IELTS_REGEX = re.compile(r"\bIELTS\s*(?:score\s*)?(\d(?:\.\d)?)\b",re.IGNORECASE)

TOEFL_REGEX = re.compile(r"\bTOEFL(?:\s*(?:iBT|CBT|PBT))?\s*(\d{2,3})\b",re.IGNORECASE)

CAMBRIDGE_REGEX = re.compile(
    r"\b("
    r"CPE|CAE|FCE|PET|KET|"
    r"BEC(?:\s*(?:Preliminary|Vantage|Higher))?|"
    r"Business English Certificate(?:\s*(?:Preliminary|Vantage|Higher))?|"
    r"Certificate of Proficiency in English|"
    r"Certificate in Advanced English|"
    r"First Certificate(?: in English)?|"
    r"Preliminary English Test|"
    r"Key English Test"
    r")\b",
    re.IGNORECASE)

LINGUASKILL_REGEX = re.compile(r"\bLinguaskill\b[^\d]{0,10}(\d{2,3})\b", re.IGNORECASE)

EFSET_REGEX = re.compile(r"\bEF\s*SET\b[^\d]{0,15}(\d{2,3})\b", re.IGNORECASE)

# ----------------------------------------------------------
# English fallback (conservative heuristic)
# ----------------------------------------------------------

ENGLISH_ANCHOR_WORDS = [
    # CV section headers / structure
    "experience", "education", "skills", "projects", "certifications",
    "summary", "profile", "objective", "achievements",

    # Employment wording
    "internship", "responsibilities", "developed", "implemented",
    "managed", "led", "designed", "optimized",

    # Academic / timeline wording
    "present", "current", "expected", "projected",

    # Common CV meta
    "contact", "phone", "email", "linkedin",]

ENGLISH_ANCHOR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in ENGLISH_ANCHOR_WORDS) + r")\b",re.IGNORECASE)

# ----------------------------------------------------------
# Strong separators that likely indicate end of language entry
# ----------------------------------------------------------
STRONG_SEPARATORS = re.compile(r"[\n;|•]")

# ----------------------------------------------------------
# Helpers for Language
# ----------------------------------------------------------
def cefr_to_level(cefr: str) -> Optional[str]:
    """
    Convert a CEFR level (A1–C2) into a normalized internal proficiency category.

    The system internally standardizes all language proficiency representations
    into four discrete categories:

        • Beginner
        • Intermediate
        • Advanced
        • Native

    Parameters
    ----------
    cefr : str
        CEFR level string (e.g., "A1", "B2", "C1", "C2").

    Returns
    -------
    Optional[str]
        Normalized proficiency category or None if not recognized.

    Notes
    -----
    C2 is mapped to "Advanced" rather than "Native", as native proficiency
    is treated as a separate semantic category in this system.
    """
    mapping = {
        "A1": "Beginner",
        "A2": "Beginner",
        "B1": "Intermediate",
        "B2": "Intermediate",
        "C1": "Advanced",
        "C2": "Advanced",
    }
    return mapping.get(cefr)

def ielts_to_cefr(score: float) -> Optional[str]:
    """
    Convert an IELTS score into the corresponding CEFR level.

    This mapping follows commonly accepted equivalence tables
    between IELTS (0–9 scale) and CEFR (A1–C2).

    Parameters
    ----------
    score : float
        IELTS band score (e.g., 6.5, 7.0, 8.5).

    Returns
    -------
    Optional[str]
        Corresponding CEFR level (e.g., "B2", "C1") or None if invalid.

    Notes
    -----
    This function assumes standard IELTS Academic/General scoring.
    """
    if score >= 8.5:
        return "C2"
    if score >= 7.0:
        return "C1"
    if score >= 5.5:
        return "B2"
    if score >= 4.0:
        return "B1"
    if score >= 3.0:
        return "A2"
    if score >= 2.0:
        return "A1"
    return None

def toefl_to_cefr(score: int) -> Optional[str]:
    """
    Convert a TOEFL score into the corresponding CEFR level.

    Supports:
        • TOEFL iBT (0–120)
        • TOEFL CBT (0–300)
        • TOEFL PBT (310–677)

    Parameters
    ----------
    score : int
        TOEFL score extracted from text.

    Returns
    -------
    Optional[str]
        Corresponding CEFR level or None if score is outside
        recognized ranges.

    Notes
    -----
    The function infers the TOEFL version based on score range.
    """

    # iBT scale (0–120)
    if score <= 120:
        if score >= 110:
            return "C2"
        if score >= 95:
            return "C1"
        if score >= 72:
            return "B2"
        if score >= 42:
            return "B1"
        if score >= 30:
            return "A2"
        return "A1"

    # CBT scale (0–300)
    if score <= 300:
        if score >= 270:
            return "C1"
        if score >= 227:
            return "B2"
        if score >= 163:
            return "B1"
        return "A2"

    # PBT scale (310–677)
    if score <= 677:
        if score >= 637:
            return "C1"
        if score >= 567:
            return "B2"
        if score >= 487:
            return "B1"
        return "A2"

    return None

def cambridge_to_cefr(name: str) -> Optional[str]:
    """
    Convert a Cambridge English qualification into the corresponding CEFR level.

    Supported certifications:
        • CPE → C2
        • CAE → C1
        • FCE → B2
        • PET → B1
        • KET → A2
        • BEC Preliminary → B1
        • BEC Vantage → B2
        • BEC Higher → C1

    Parameters
    ----------
    name : str
        Certification name or abbreviation extracted from text.

    Returns
    -------
    Optional[str]
        Corresponding CEFR level or None if not recognized.
    """

    name = name.lower().strip()

    mapping = {
        # Cambridge General English
        "cpe": "C2",
        "certificate of proficiency in english": "C2",

        "cae": "C1",
        "certificate in advanced english": "C1",

        "fce": "B2",
        "first certificate": "B2",
        "first certificate in english": "B2",

        "pet": "B1",
        "preliminary english test": "B1",

        "ket": "A2",
        "key english test": "A2",

        # BEC (Business English Certificates)
        "bec preliminary": "B1",
        "business english certificate preliminary": "B1",

        "bec vantage": "B2",
        "business english certificate vantage": "B2",

        "bec higher": "C1",
        "business english certificate higher": "C1",

        # If only "BEC" appears without level, assume intermediate baseline
        "bec": "B2",
    }

    return mapping.get(name)

def linguaskill_to_cefr(score: int) -> Optional[str]:
    """
    Convert a Linguaskill score (100–180 scale) into CEFR level.
    """

    if score >= 180:
        return "C2"
    if score >= 170:
        return "C1"
    if score >= 160:
        return "B2"
    if score >= 140:
        return "B1"
    if score >= 120:
        return "A2"
    return "A1"

def efset_to_cefr(score: int) -> Optional[str]:
    """
    Convert an EF SET score (0–100 scale) into CEFR level.
    """

    if score >= 90:
        return "C2"
    if score >= 75:
        return "C1"
    if score >= 60:
        return "B2"
    if score >= 45:
        return "B1"
    if score >= 30:
        return "A2"
    return "A1"

def _search_level_in_context(context: str,*,base_offset: int,reverse: bool = False) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """
    Search for language proficiency indicators within a bounded text segment.

    This function scans a context substring (either to the right or left of a
    detected language mention) and identifies possible proficiency signals,
    including:

        • Standardized test scores (IELTS, TOEFL, Linguaskill, EF SET)
        • Cambridge English certifications
        • Free-text descriptors (e.g., "fluent", "B2", "native")

    All detected candidates are ranked based on textual proximity to the
    language mention. The closest match is returned.

    Parameters
    ----------
    context : str
        Text segment in which proficiency indicators are searched.
    base_offset : int
        Reference offset used to compute relative distance.
    reverse : bool, optional
        If True, distances are computed from the end of the context
        (used for left-side searches).

    Returns
    -------
    Optional[Tuple[Optional[str], Optional[str]]]
        A tuple containing:
            • normalized proficiency level (Beginner / Intermediate / Advanced / Native)
            • raw matched expression (e.g., "IELTS 7.0", "C1", "fluent")
        Returns None if no indicator is found.

    Notes
    -----
    Distance-based ranking ensures that the most contextually relevant
    proficiency descriptor is selected when multiple indicators are present.
    """
    candidates: List[Tuple[int, str, str]] = []

    # ---- IELTS ----
    for m in IELTS_REGEX.finditer(context):
        try:
            score = float(m.group(1))
            cefr = ielts_to_cefr(score)
            if not cefr:
                continue

            level = cefr_to_level(cefr)
            if level is None:
                continue

            dist = abs((base_offset - m.start()) if reverse else m.start())
            candidates.append((dist, level, f"IELTS {score}"))
        except Exception:
            continue

    # ---- TOEFL ----
    for m in TOEFL_REGEX.finditer(context):
        try:
            score = int(m.group(1))
            cefr = toefl_to_cefr(score)
            if not cefr:
                continue

            level = cefr_to_level(cefr)
            if level is None:
                continue

            dist = abs((base_offset - m.start()) if reverse else m.start())
            candidates.append((dist, level, f"TOEFL {score}"))
        except Exception:
            continue

    # ---- Linguaskill ----
    for m in LINGUASKILL_REGEX.finditer(context):
        try:
            score = int(m.group(1))
            cefr = linguaskill_to_cefr(score)
            if not cefr:
                continue

            level = cefr_to_level(cefr)
            if level is None:
                continue

            dist = abs((base_offset - m.start()) if reverse else m.start())
            candidates.append((dist, level, f"Linguaskill {score}"))
        except Exception:
            continue

    # ---- EF SET ----
    for m in EFSET_REGEX.finditer(context):
        try:
            score = int(m.group(1))
            cefr = efset_to_cefr(score)
            if not cefr:
                continue

            level = cefr_to_level(cefr)
            if level is None:
                continue

            dist = abs((base_offset - m.start()) if reverse else m.start())
            candidates.append((dist, level, f"EF SET {score}"))
        except Exception:
            continue

    # ---- Cambridge ----
    for m in CAMBRIDGE_REGEX.finditer(context):
        cefr = cambridge_to_cefr(m.group(1))
        if not cefr:
            continue

        level = cefr_to_level(cefr)
        if level is None:
            continue

        dist = abs((base_offset - m.start()) if reverse else m.start())
        candidates.append((dist, level, m.group(1)))

    # ---- Free text tokens ----
    for raw_term, normalized_level in LEVEL_LOOKUP.items():
        for m in re.finditer(rf"\b{re.escape(raw_term)}\b", context, re.IGNORECASE):
            dist = abs((base_offset - m.start()) if reverse else m.start())
            candidates.append((dist, normalized_level, raw_term))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, level, raw = candidates[0]

    return level, raw

def _truncate_context(text: str, start: int, end: int) -> str:
    """
    Extract a bounded substring and prevent cross-language contamination.

    This function isolates a portion of text around a detected language
    mention and truncates it at:

        • strong structural separators (newline, semicolon, bullet, pipe)
        • the appearance of another language name

    The goal is to prevent proficiency indicators from "leaking" across
    separate language entries in structured CV lists.

    Parameters
    ----------
    text : str
        Full CV text.
    start : int
        Starting index of the context window.
    end : int
        Ending index of the context window.

    Returns
    -------
    str
        Truncated context segment considered safe for proficiency detection.
    """
    segment = text[start:end]

    # Stop at strong separator
    sep_match = STRONG_SEPARATORS.search(segment)
    if sep_match:
        segment = segment[:sep_match.start()]

    # Stop if another language appears
    lang_match = LANG_PATTERN.search(segment)
    if lang_match:
        segment = segment[:lang_match.start()]

    return segment

def _find_level_near_language(text: str,
    lang_span: tuple[int, int],
    window: int = 120,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the most relevant proficiency level near a detected language mention.

    This function performs a bidirectional contextual search:

        1. It first inspects the right-hand context (higher priority).
        2. If no level is found, it inspects the left-hand context.
        3. Both contexts are truncated at strong separators or the next language.

    The closest proficiency indicator is selected based on distance ranking.

    Parameters
    ----------
    text : str
        Full CV text.
    lang_span : tuple[int, int]
        Character span (start, end) of the detected language match.
    window : int, optional
        Maximum number of characters to inspect on each side.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        A tuple containing:
            • normalized proficiency category
            • raw detected expression

    Notes
    -----
    Right-side matches are always prioritized over left-side matches.
    If no proficiency indicator is found, (None, None) is returned.
    """

    start, end = lang_span

    # -------------------------
    # RIGHT CONTEXT (priority)
    # -------------------------
    right_limit = min(len(text), end + window)
    right_context = _truncate_context(text, end, right_limit)

    right_candidate = _search_level_in_context(right_context, base_offset=0)

    if right_candidate:
        return right_candidate

    # -------------------------
    # LEFT CONTEXT
    # -------------------------
    left_start = max(0, start - window)
    left_context = _truncate_context(text, left_start, start)

    # Reverse search logic by computing distance from end of left_context
    left_candidate = _search_level_in_context(
        left_context,
        base_offset=len(left_context),
        reverse=True
    )

    if left_candidate:
        return left_candidate

    return None, None

def is_probably_english_text(
    text: str,
    *,
    min_words: int = 80,
    min_unique_anchors: int = 4,
) -> bool:
    """
    Heuristically determine whether a CV is primarily written in English.

    This conservative fallback mechanism is triggered only when no explicit
    "English" language entry is detected. It relies on:

        • Minimum document length
        • Presence of multiple distinct English CV anchor terms

    The approach is designed to minimize false positives while accounting
    for CVs from English-speaking countries where language proficiency
    is often implicit.

    Parameters
    ----------
    text : str
        Raw CV text (OCR or parsed).
    min_words : int, optional
        Minimum token count required to consider detection.
    min_unique_anchors : int, optional
        Minimum number of distinct English anchor words required.

    Returns
    -------
    bool
        True if the text is strongly indicative of English usage,
        otherwise False.
    """
    if not text:
        return False

    # Basic length guard to avoid false positives on short snippets
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < min_words:
        return False

    hits = {m.group(1).lower() for m in ENGLISH_ANCHOR_PATTERN.finditer(text)}
    return len(hits) >= min_unique_anchors

# ----------------------------------------------------------
# Language functions
# ----------------------------------------------------------
class LanguageInfo(TypedDict):
    """
    Typed structure representing extracted language information.

    Fields
    ------
    language : str
        Canonical language name.
    level : Optional[str]
        Normalized proficiency category
        (Beginner / Intermediate / Advanced / Native).
    level_raw : Optional[str]
        Original textual representation detected in the CV
        (e.g., "C1", "IELTS 7.5", "fluent", "native").
    """
    language: str
    level: Optional[str]          # Beginner / Intermediate / Advanced / Native
    level_raw: Optional[str]      # A1, fluent, native, etc.

def _extract_languages_full(text: str) -> List[LanguageInfo]:
    """
    Extract structured language information from CV text.

    The extraction pipeline follows these steps:

        1. Identify explicit language mentions using a predefined vocabulary.
        2. Search for nearby proficiency indicators using contextual ranking.
        3. Normalize proficiency representations into standardized categories.
        4. Deduplicate entries while preserving the highest detected level.
        5. Optionally append English if the document is strongly indicative
        of English usage but no explicit English entry is present.

    The English fallback:
        • Never overrides explicit entries.
        • Never assigns an assumed proficiency level.
        • Only appends "English" with unknown level when appropriate.

    Parameters
    ----------
    text : str
        Raw CV text.

    Returns
    -------
    List[LanguageInfo]
        Sorted list of detected languages with structured proficiency data.

    Notes
    -----
    When multiple proficiency indicators are detected for the same language,
    the highest-ranked level is retained according to LEVEL_RANK.
    """

    if not text:
        return []

    # --------------------------------------------------
    # 1) Extract explicit language mentions (+ nearby levels)
    # --------------------------------------------------
    results: List[LanguageInfo] = []
    text_lower = text.lower()

    for match in LANG_PATTERN.finditer(text_lower):
        lang = LANG_LOOKUP[match.group(1).lower()]
        level, raw = _find_level_near_language(text, match.span())

        results.append({
            "language": lang,
            "level": level,
            "level_raw": raw,
        })

    # --------------------------------------------------
    # 2) Deduplicate, keeping the highest proficiency level per language
    # --------------------------------------------------
    unique: Dict[str, LanguageInfo] = {}

    for item in results:
        lang = item["language"]

        if lang not in unique:
            unique[lang] = item
            continue

        old_level = unique[lang]["level"]
        new_level = item["level"]

        # Keep the entry with the highest proficiency rank (if any)
        if new_level and (
            not old_level
            or LEVEL_RANK.get(new_level, -1) > LEVEL_RANK.get(old_level, -1)
        ):
            unique[lang] = item

    deduped = sorted(unique.values(), key=lambda x: x["language"])

    # --------------------------------------------------
    # 3) English fallback (append-only)
    # --------------------------------------------------
    english_present = any(item["language"] == "English" for item in deduped)

    # If English is not explicitly listed but the document is likely English,
    # append English with unknown level.
    if not english_present and is_probably_english_text(text):
        deduped.append({
            "language": "English",
            "level": None,
            "level_raw": None,
        })
        deduped = sorted(deduped, key=lambda x: x["language"])

    return deduped

def extract_languages(text: str) -> List[str]:
    """
    Extract a unique list of detected languages without level details.

    Parameters
    ----------
    text : str
        Raw CV text.

    Returns
    -------
    List[str]
        Sorted list of language names.
    """
    return [item["language"] for item in _extract_languages_full(text)]

def extract_languages_details(text: str) -> List[LanguageInfo]:
    """
    Extract structured language information including proficiency levels.

    Parameters
    ----------
    text : str
        Raw CV text.

    Returns
    -------
    List[LanguageInfo]
        List of language entries with normalized and raw proficiency data.

    Notes
    -----
    This function preserves both:
        • normalized proficiency category
        • original detected expression (e.g., "IELTS 7.0")
    """
    return _extract_languages_full(text)


# ==========================================================
# Regex patterns for email and phone number
# ==========================================================
EMAIL_REGEX = re.compile(
    r"""
    (?<![\w.+-@])           
    [A-Za-z0-9._%+-]+
    @[A-Za-z0-9.-]+
    \.[A-Za-z]{2,}
    """,
    re.VERBOSE | re.IGNORECASE,
)

EMAIL_PROVIDERS = {
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "yahoo.com",
    "icloud.com",
    "live.com",
}

PHONE_REGEX = re.compile(
    r"""
    (?:
        (?:(?:\+|00)\d{1,3}[\s\-]?)?      # optional country code
        (?:\(?\d{2,3}\)?[\s\-]?)?         # optional area code
        \d{3}[\s\-]?\d{3}[\s\-]?\d{3,4}    # main number
    )
    """,
    re.VERBOSE,
)

# ==========================================================
# Regex Sites
# ==========================================================
LINKEDIN_REGEX = re.compile(r"(https?://)?(www\.)?linkedin\.com/[^\s]+",re.IGNORECASE,)

GITHUB_REGEX = re.compile(r"(https?://)?(www\.)?github\.com/[^\s]+",re.IGNORECASE,)

URL_REGEX = re.compile(r"""\b(?:https?://|www\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[^\s]*)?""",re.VERBOSE | re.IGNORECASE,)


# ----------------------------------------------------------
# Sites functions
# ----------------------------------------------------------
def extract_all_websites(text: str) -> List[str]:
    if not text:
        return []

    urls = set()

    # Extrair emails primeiro para remover depois
    emails = extract_all(EMAIL_REGEX, text)
    email_domains = {
        e.split("@")[1].lower()
        for e in emails
        if "@" in e
    }

    for match in URL_REGEX.finditer(text):
        url = match.group(0).strip()

        # Normalizar
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        domain = re.sub(r"https?://(www\.)?", "", url)
        domain = domain.split("/")[0].lower()

        # Ignorar se for domínio de email extraído
        if domain in email_domains:
            continue

        # Ignorar se for provider genérico
        if domain in EMAIL_PROVIDERS:
            continue

        # Ignorar domínios muito curtos (ex: www.)
        if len(domain.split(".")) < 2:
            continue

        urls.add(url)

    return sorted(urls)


# ==========================================================
# Regex Adress
# ==========================================================
ADDRESS_LINE_REGEX = re.compile(
    r"""
    (?:^|\n)                                   # start of line
    (?:Address|Morada|Location|Residence|Home)      # explicit label
    \s*[:\-]\s*
    ([^\n]{5,120})                             # single-line address
    """,
    re.IGNORECASE | re.VERBOSE,
)

ADDRESS_KEYWORDS = {"address", "location", "home", "residence","morada", "endereço", "based", "living"}


# ----------------------------------------------------------
# Helpers for Adress
# ----------------------------------------------------------
def _fallback_geo(text: str) -> Optional[str]:
    """
    Conservative geographic fallback.

    Priority:
    1) Capital detected → "Lisbon, Portugal"
    2) Country detected → "Portugal"
    3) None
    """

    if not text:
        return None

    text_lower = text.lower()


    # ---------- 1) Detect capital ----------
    for country, capital in COUNTRIES_AND_CAPITALS.items():
        if not capital:
            continue

        if re.search(rf"\b{re.escape(capital.lower())}\b", text_lower):
            return f"{capital}, {country}"

    # ---------- 2) Detect country ----------
    for country in COUNTRIES_AND_CAPITALS.keys():
        if country.lower() in text_lower:
            return country

    return None


# ----------------------------------------------------------
# Adress functions
# ----------------------------------------------------------
def extract_address(text: str) -> Optional[str]:
    """
    Strict + semantic geographic extraction.

    Priority:
    1) Explicit labeled address
    2) Capital → "Lisbon, Portugal"
    3) Country → "Portugal"
    4) None
    """

    if not text:
        return None

    text = clean_text(text)

    # ---------- 1) Explicit address ----------
    match = ADDRESS_LINE_REGEX.search(text)
    if match:
        candidate = match.group(1).strip()

        has_postal = re.search(r"\b\d{4,6}(-\d{3})?\b", candidate)
        candidate_lower = candidate.lower()
        has_country = any(re.search(rf"\b{re.escape(c.name.lower())}\b", candidate_lower) for c in pycountry.countries)


        if has_postal or has_country:
            return candidate

    # ---------- 2) Geographic fallback ----------
    geo = _fallback_geo(text)
    if geo:
        return geo

    return None


# ----------------------------------------------------------
# Regex experience
# ----------------------------------------------------------
EXPLICIT_YEARS_REGEX = re.compile(
    r"""
    (?:
        # optional qualifiers before the number
        less\ than|
        under|
        more\ than|
        over|
        at\ least|
        minimum|
        min\.?|
        up\ to|
        approx\.?|
        around|
        about
    )?
    \s*

    (\d{1,2})        # number of years
    \+?              # optional plus (e.g., 5+)

    \s*
    (?:
        years?|yrs?|y
    )

    (?:
        \s+of\s+experience|
        \s+experience|
        \s+exp\.?
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

DATE_RANGE_REGEX = re.compile(
    r"""
    (?:
        # optional prefixes
        (?:since|from|year\s+)?\s*
    )

    # ---------------- START DATE ----------------
    (?:
        (?:\d{1,2}[/-]){0,2}      # optional day/month
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
           January|February|March|April|June|July|August|
           September|October|November|December)?\s*
        (\d{2,4})                 # start year
    )

    \s*
    (?:-|–|—|~|\bto\b|\buntil\b|\btill\b|\bthrough\b|→|~)
    \s*

    # ---------------- END DATE ----------------
    (?:
        (?:\d{1,2}[/-]){0,2}
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
           January|February|March|April|June|July|August|
           September|October|November|December)?\s*
        (\d{2,4}|Present|Current|Today|Now)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SINCE_REGEX = re.compile(
    r"""
    (?:since|from)\s+
    (\d{2,4})
    """,
    re.IGNORECASE | re.VERBOSE,
)

EDU_KEYWORDS = {"education", "bachelor", "master","degree", "course", "student", "academic"}

WORK_KEYWORDS = {"experience", "work", "employment", "career", "history", "intern", "job", "company"}


# ----------------------------------------------------------
# Helpers for Experience
# ----------------------------------------------------------
def _normalize_year(y: str) -> Optional[int]:
    """Converts text year to valid integer."""
    y = y.lower()

    if y in {"present", "current", "today", "now", "nowadays", "contemporary"}:
        return CURRENT_YEAR

    try:
        year = int(y)
    except ValueError:
        return None


    # tratar anos curtos tipo 23 → 2023
    if year < 100:
        year += 2000

    if 1950 <= year <= CURRENT_YEAR + 1:
        return year

    return None

def _context_has_keywords(text: str, pos: int, direction: str, window_words: int) -> str:
    """
    Search for keywords on the right or left.
    Returns: "edu", "work", or "".
    """
    words = re.split(r"\s+", text)

    # encontrar índice aproximado da posição
    char_count = 0
    idx = 0
    for i, w in enumerate(words):
        char_count += len(w) + 1
        if char_count >= pos:
            idx = i
            break

    if direction == "right":
        segment = words[idx : idx + window_words]
    else:
        segment = words[max(0, idx - window_words) : idx]

    segment_set = {
        re.sub(r"[^\w]", "", w.lower())
        for w in segment
        if w}


    if segment_set & EDU_KEYWORDS:
        return "edu"
    if segment_set & WORK_KEYWORDS:
        return "work"
    return ""

def _merge_intervals(intervals: List[tuple[int, int]]) -> int:
    """Combine overlapping intervals and return total duration in years."""
    if not intervals:
        return 0

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        # overlap → extend interval
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    # total duration
    return sum(max(0, end - start) for start, end in merged)


# ----------------------------------------------------------
# Experience functions
# ----------------------------------------------------------
def extract_experience_years(text: str) -> Optional[int]:
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
           • short right-side context (≈10 words)
           • long left-side context (≈400 words, section titles)

    5) Interval merging
       - Prevents double counting overlapping jobs

    Returns
    -------
    Optional[int]
        Total estimated years of professional experience.
        Returns None if no reliable estimation is possible.
    """

    # Safety check
    if not text:
        return None

    text = clean_text(text)

    # ------------------------------------------------------
    # 1) Explicit experience mentions (e.g., "5 years")
    # ------------------------------------------------------
    explicit_years = [
        int(m.group(1))
        for m in EXPLICIT_YEARS_REGEX.finditer(text)
        if 0 < int(m.group(1)) <= 50
    ]

    # ------------------------------------------------------
    # 2) Closed date ranges (start → end)
    # ------------------------------------------------------
    intervals: List[tuple[int, int]] = []

    for m in DATE_RANGE_REGEX.finditer(text):

        start_raw, end_raw = m.group(1), m.group(2)

        start = _normalize_year(start_raw)
        end = _normalize_year(end_raw)

        # Discard invalid or reversed ranges
        if not start or not end or end <= start:
            continue

        pos = m.start()

        # ---- Context classification ----
        right_ctx = _context_has_keywords(text, pos, "right", 10)

        # Ignore education periods
        if right_ctx == "edu":
            continue

        # Accept if clearly work-related
        if right_ctx == "work":
            intervals.append((start, end))
            continue

        # Fallback: inspect section titles on the left
        left_ctx = _context_has_keywords(text, pos, "left", 400)

        if left_ctx == "work":
            intervals.append((start, end))
            continue

    # ------------------------------------------------------
    # 2b) Open intervals like "since 2020"
    # ------------------------------------------------------
    for m in SINCE_REGEX.finditer(text):

        start = _normalize_year(m.group(1))
        if not start:
            continue

        pos = m.start()

        right_ctx = _context_has_keywords(text, pos, "right", 10)

        # Ignore education context
        if right_ctx == "edu":
            continue

        left_ctx = _context_has_keywords(text, pos, "left", 400)

        # Accept only if work-related and avoid duplicates
        if right_ctx == "work" or (left_ctx == "work" and right_ctx != "edu"):
            interval = (start, CURRENT_YEAR)
            if interval not in intervals:
                intervals.append(interval)

    # ------------------------------------------------------
    # 3) Merge overlapping intervals and sum durations
    # ------------------------------------------------------
    total_years = _merge_intervals(intervals)

    # Reject unrealistic values
    if total_years > 50:
        return None

    # ------------------------------------------------------
    # 4) Fallback to explicit mentions if no intervals found
    # ------------------------------------------------------
    if total_years == 0:
        if explicit_years:
            return max(explicit_years)
        if intervals:
            return _merge_intervals(intervals)

    if total_years < 0:
        return None
    else:
        return total_years


# ==========================================================
# Text loading (Memory-based)
# ==========================================================
def clean_text(text: str) -> str:
    """Remove common OCR artifacts and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t\uf0b7\u2022*]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def normalize_text_input(text_input: Union[str, Mapping[str, Any], Sequence[Any]]) -> str:
    """
    Normalizes text input to a single string.
    
    Handles:
    - String: Returns as-is
    - Dict with 'text' key: Extracts text value
    - List of strings: Joins with newlines
    """
    if isinstance(text_input, str):
        return text_input
    elif isinstance(text_input, dict):
        return text_input.get("text", "")
    elif isinstance(text_input, list):
        # If list of dicts, extract text from each
        if text_input and isinstance(text_input[0], dict):
            return "\n\n".join([item.get("text", "") for item in text_input])
        # If list of strings, join them
        return "\n\n".join([str(item) for item in text_input])
    else:
        raise TypeError(f"Expected str, dict, or list. Got {type(text_input)}")

def normalize_email(email: str) -> str:
    return email.lower().strip()

def normalize_phone(phone: str, default_region: Optional[str] = None) -> Optional[str]:
    """
    Professional phone normalization using Google's phonenumbers library.

    - Validates number structure
    - Preserves international prefix
    - Converts to E.164 format
    - Does NOT blindly assume a country
    """

    if not phone:
        return None

    try:
        # Parse with optional region (only used if no prefix exists)
        parsed = phonenumbers.parse(phone, default_region if default_region else None)

        # Validate number
        if not phonenumbers.is_valid_number(parsed):
            return None

        # Return international E.164 format
        return phonenumbers.format_number(
            parsed,
            phonenumbers.PhoneNumberFormat.E164
        )

    except NumberParseException:
        return None
    
# ==========================================================
# General extraction helpers
# ==========================================================
def extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
    """Retorn the first ocorrence of a patern.."""
    match = pattern.search(text)
    return match.group(0).strip() if match else None

def extract_all(pattern: Pattern[str], text: str) -> List[str]:
    """Retorn all unique ocorrences of a patern."""
    matches = pattern.findall(text)
    cleaned = []

    for m in matches:
        if isinstance(m, tuple):
            m = "".join(m)

        m = m.strip()
        if m and m not in cleaned:
            cleaned.append(m)

    return cleaned

    
# ==========================================================
# Core extraction
# ==========================================================
class ContactInfo(TypedDict):
    email: Optional[str]
    all_emails: List[str]
    phone: Optional[str]
    all_phones: List[str]
    address: Optional[str]
    experience_years: Optional[int]
    languages: List[str]
    languages_details: List[LanguageInfo]
    linkedin: Optional[str]
    github: Optional[str]
    all_websites: List[str]

def extract_contact_info(text: str) -> ContactInfo:
    emails = [normalize_email(e) for e in extract_all(EMAIL_REGEX, text)]
    address = extract_address(text)

    raw_phones = extract_all(PHONE_REGEX, text)
    normalized_phones = []

    # Detect region from address if possible
    region_code: Optional[str] = None
    if address:
        for country in pycountry.countries:
            if country.name.lower() in address.lower():
                region_code = country.alpha_2
                break

    international_found = False

    for p in raw_phones:
        # If phone has + or 00 → parse without region assumption
        if p.strip().startswith(("+", "00")):
            norm = normalize_phone(p)
            if norm:
                international_found = True
                normalized_phones.append(norm)
        else:
            # Try with detected region
            norm = normalize_phone(p, default_region=region_code)
            if norm:
                normalized_phones.append(norm)

    # If at least one international number exists → keep only international
    if international_found:
        phones = [p for p in normalized_phones if p.startswith("+")]
    else:
        phones = normalized_phones

    linkedin = extract_first(LINKEDIN_REGEX, text)
    github = extract_first(GITHUB_REGEX, text)
    
    experience_years = extract_experience_years(text)

    languages_details = extract_languages_details(text)
    languages = [l["language"] for l in languages_details]

    all_websites = extract_all_websites(text)

    return {
        "email": emails[0] if emails else None,
        "all_emails": emails,
        "phone": phones[0] if phones else None,
        "all_phones": phones,
        "address": address,
        "experience_years": experience_years,
        "languages": languages,
        "languages_details": languages_details,
        "linkedin": linkedin,
        "github": github,
        "all_websites": all_websites,
    }


# ==========================================================
# Dataset processing (Memory-based)
# ==========================================================
def process_texts(
    texts: Union[List[str], List[Dict], Dict[str, str]],
) -> pd.DataFrame:
    """
    Processes text data in memory and extracts contact info.
    
    Args:
        texts: One of:
            - List of strings
            - List of dicts with 'text' key
            - Dict mapping identifiers to text
    
    Returns:
        DataFrame with extracted contact information.
    
    Example:
        >>> texts = ["Email: john@example.com", "Phone: +1-555-1234"]
        >>> df = process_texts(texts)
    """
    
    if isinstance(texts, dict) and not ("text" in texts and isinstance(texts["text"], str)):
        # Dict mapping identifiers to text
        text_items = texts.items()
    elif isinstance(texts, list):
        # List of strings or dicts - assign numeric IDs
        text_items = [(i, item) for i, item in enumerate(texts)]
    else:
        raise ValueError("texts must be a list or dict")
    
    rows = []
    
    for identifier, text_data in text_items:
        try:
            # Normalize text input
            text = normalize_text_input(text_data)
            
            # Extract contact info
            info = extract_contact_info(text)
            
            rows.append({
                "id": identifier,
                **info,
            })
        
        except Exception as e:
            rows.append({
                "id": identifier,
                "error": str(e),
            })
    
    return pd.DataFrame(rows)


# ==========================================================
# CLI execution
# ==========================================================

if __name__ == "__main__":
    # Example: Memory-based usage
    # Process text from OCR pipeline
    sample_texts = [
        "Email: john.doe@example.com\nPhone: +1-555-1234\nLinkedIn: https://linkedin.com/in/johndoe",
        "Contact: jane_smith@company.org\nPhone: (555) 987-6543",
    ]
    
    df = process_texts(sample_texts)
    print(df.head())
    
    # Alternative: Single text
    # result = extract_contact_info("Email: test@example.com")
    # print(result)
