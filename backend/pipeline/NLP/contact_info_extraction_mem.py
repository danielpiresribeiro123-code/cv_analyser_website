import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, TypedDict

from datetime import datetime
CURRENT_YEAR = datetime.now().year


# ----------------------------------------------------------
# Load countries once (fast lookup)
# ----------------------------------------------------------
import pycountry
CAPITALS = {'Afghanistan': 'Kabul', 'Aland Islands': 'Mariehamn', 'Albania': 'Tirana', 'Algeria': 'Algiers', 'American Samoa': 'Pago Pago', 'Andorra': 'Andorra la Vella', 'Angola': 'Luanda', 'Anguilla': 'The Valley', 'Antarctica': '', 'Antigua And Barbuda': "St. John's", 'Argentina': 'Buenos Aires', 'Armenia': 'Yerevan', 'Aruba': 'Oranjestad', 'Australia': 'Canberra', 'Austria': 'Vienna', 'Azerbaijan': 'Baku', 'Bahamas The': 'Nassau', 'Bahrain': 'Manama', 'Bangladesh': 'Dhaka', 'Barbados': 'Bridgetown', 'Belarus': 'Minsk', 'Belgium': 'Brussels', 'Belize': 'Belmopan', 'Benin': 'Porto-Novo', 'Bermuda': 'Hamilton', 'Bhutan': 'Thimphu', 'Bolivia': 'Sucre', 'Bosnia and Herzegovina': 'Sarajevo', 'Botswana': 'Gaborone', 'Bouvet Island': '', 'Brazil': 'Brasilia', 'British Indian Ocean Territory': 'Diego Garcia', 'Brunei': 'Bandar Seri Begawan', 'Bulgaria': 'Sofia', 'Burkina Faso': 'Ouagadougou', 'Burundi': 'Bujumbura', 'Cambodia': 'Phnom Penh', 'Cameroon': 'Yaounde', 'Canada': 'Ottawa', 'Cape Verde': 'Praia', 'Cayman Islands': 'George Town', 'Central African Republic': 'Bangui', 'Chad': "N'Djamena", 'Chile': 'Santiago', 'China': 'Beijing', 'Christmas Island': 'Flying Fish Cove', 'Cocos (Keeling) Islands': 'West Island', 'Colombia': 'Bogota', 'Comoros': 'Moroni', 'Congo': 'Brazzaville', 'Congo The Democratic Republic Of The': 'Kinshasa', 'Cook Islands': 'Avarua', 'Costa Rica': 'San Jose', "Cote D'Ivoire (Ivory Coast)": 'Yamoussoukro', 'Croatia': 'Zagreb', 'Cuba': 'Havana', 'Cyprus': 'Nicosia', 'Czech Republic': 'Prague', 'Denmark': 'Copenhagen', 'Djibouti': 'Djibouti', 'Dominica': 'Roseau', 'Dominican Republic': 'Santo Domingo', 'East Timor': 'Dili', 'Ecuador': 'Quito', 'Egypt': 'Cairo', 'El Salvador': 'San Salvador', 'Equatorial Guinea': 'Malabo', 'Eritrea': 'Asmara', 'Estonia': 'Tallinn', 'Ethiopia': 'Addis Ababa', 'Falkland Islands': 'Stanley', 'Faroe Islands': 'Torshavn', 'Fiji Islands': 'Suva', 'Finland': 'Helsinki', 'France': 'Paris', 'French Guiana': 'Cayenne', 'French Polynesia': 'Papeete', 'French Southern Territories': 'Port-aux-Francais', 'Gabon': 'Libreville', 'Gambia The': 'Banjul', 'Georgia': 'Tbilisi', 'Germany': 'Berlin', 'Ghana': 'Accra', 'Gibraltar': 'Gibraltar', 'Greece': 'Athens', 'Greenland': 'Nuuk', 'Grenada': "St. George's", 'Guadeloupe': 'Basse-Terre', 'Guam': 'Hagatna', 'Guatemala': 'Guatemala City', 'Guernsey and Alderney': 'St Peter Port', 'Guinea': 'Conakry', 'Guinea-Bissau': 'Bissau', 'Guyana': 'Georgetown', 'Haiti': 'Port-au-Prince', 'Heard and McDonald Islands': '', 'Honduras': 'Tegucigalpa', 'Hong Kong S.A.R.': 'Hong Kong', 'Hungary': 'Budapest', 'Iceland': 'Reykjavik', 'India': 'New Delhi', 'Indonesia': 'Jakarta', 'Iran': 'Tehran', 'Iraq': 'Baghdad', 'Ireland': 'Dublin', 'Israel': 'Jerusalem', 'Italy': 'Rome', 'Jamaica': 'Kingston', 'Japan': 'Tokyo', 'Jersey': 'Saint Helier', 'Jordan': 'Amman', 'Kazakhstan': 'Astana', 'Kenya': 'Nairobi', 'Kiribati': 'Tarawa', 'Korea North': 'Pyongyang', 'Korea South': 'Seoul', 'Kuwait': 'Kuwait City', 'Kyrgyzstan': 'Bishkek', 'Laos': 'Vientiane', 'Latvia': 'Riga', 'Lebanon': 'Beirut', 'Lesotho': 'Maseru', 'Liberia': 'Monrovia', 'Libya': 'Tripolis', 'Liechtenstein': 'Vaduz', 'Lithuania': 'Vilnius', 'Luxembourg': 'Luxembourg', 'Macau S.A.R.': 'Macao', 'Macedonia': 'Skopje', 'Madagascar': 'Antananarivo', 'Malawi': 'Lilongwe', 'Malaysia': 'Kuala Lumpur', 'Maldives': 'Male', 'Mali': 'Bamako', 'Malta': 'Valletta', 'Man (Isle of)': 'Douglas, Isle of Man', 'Marshall Islands': 'Majuro', 'Martinique': 'Fort-de-France', 'Mauritania': 'Nouakchott', 'Mauritius': 'Port Louis', 'Mayotte': 'Mamoudzou', 'Mexico': 'Mexico City', 'Micronesia': 'Palikir', 'Moldova': 'Chisinau', 'Monaco': 'Monaco', 'Mongolia': 'Ulan Bator', 'Montenegro': 'Podgorica', 'Montserrat': 'Plymouth', 'Morocco': 'Rabat', 'Mozambique': 'Maputo', 'Myanmar': 'Nay Pyi Taw', 'Namibia': 'Windhoek', 'Nauru': 'Yaren', 'Nepal': 'Kathmandu', 'Netherlands Antilles': '', 'Netherlands The': 'Amsterdam', 'New Caledonia': 'Noumea', 'New Zealand': 'Wellington', 'Nicaragua': 'Managua', 'Niger': 'Niamey', 'Nigeria': 'Abuja', 'Niue': 'Alofi', 'Norfolk Island': 'Kingston', 'Northern Mariana Islands': 'Saipan', 'Norway': 'Oslo', 'Oman': 'Muscat', 'Pakistan': 'Islamabad', 'Palau': 'Melekeok', 'Palestinian Territory Occupied': 'East Jerusalem', 'Panama': 'Panama City', 'Papua new Guinea': 'Port Moresby', 'Paraguay': 'Asuncion', 'Peru': 'Lima', 'Philippines': 'Manila', 'Pitcairn Island': 'Adamstown', 'Poland': 'Warsaw', 'Portugal': 'Lisbon', 'Puerto Rico': 'San Juan', 'Qatar': 'Doha', 'Reunion': 'Saint-Denis', 'Romania': 'Bucharest', 'Russia': 'Moscow', 'Rwanda': 'Kigali', 'Saint Helena': 'Jamestown', 'Saint Kitts And Nevis': 'Basseterre', 'Saint Lucia': 'Castries', 'Saint Pierre and Miquelon': 'Saint-Pierre', 'Saint Vincent And The Grenadines': 'Kingstown', 'Saint-Barthelemy': 'Gustavia', 'Saint-Martin (French part)': 'Marigot', 'Samoa': 'Apia', 'San Marino': 'San Marino', 'Sao Tome and Principe': 'Sao Tome', 'Saudi Arabia': 'Riyadh', 'Senegal': 'Dakar', 'Serbia': 'Belgrade', 'Seychelles': 'Victoria', 'Sierra Leone': 'Freetown', 'Singapore': 'Singapur', 'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Solomon Islands': 'Honiara', 'Somalia': 'Mogadishu', 'South Africa': 'Pretoria', 'South Georgia': 'Grytviken', 'South Sudan': 'Juba', 'Spain': 'Madrid', 'Sri Lanka': 'Colombo', 'Sudan': 'Khartoum', 'Suriname': 'Paramaribo', 'Svalbard And Jan Mayen Islands': 'Longyearbyen', 'Swaziland': 'Mbabane', 'Sweden': 'Stockholm', 'Switzerland': 'Berne', 'Syria': 'Damascus', 'Taiwan': 'Taipei', 'Tajikistan': 'Dushanbe', 'Tanzania': 'Dodoma', 'Thailand': 'Bangkok', 'Togo': 'Lome', 'Tokelau': '', 'Tonga': "Nuku'alofa", 'Trinidad And Tobago': 'Port of Spain', 'Tunisia': 'Tunis', 'Turkey': 'Ankara', 'Turkmenistan': 'Ashgabat', 'Turks And Caicos Islands': 'Cockburn Town', 'Tuvalu': 'Funafuti', 'Uganda': 'Kampala', 'Ukraine': 'Kiev', 'United Arab Emirates': 'Abu Dhabi', 'United Kingdom': 'London', 'United States': 'Washington', 'United States Minor Outlying Islands': '', 'Uruguay': 'Montevideo', 'Uzbekistan': 'Tashkent', 'Vanuatu': 'Port Vila', 'Vatican City State (Holy See)': 'Vatican City', 'Venezuela': 'Caracas', 'Vietnam': 'Hanoi', 'British Virgin Islands': 'Road Town', 'U.S. Virgin Islands': 'Charlotte Amalie', 'USA Virgin Islands': 'Charlotte Amalie', 'Wallis And Futuna Islands': 'Mata Utu', 'Western Sahara': 'El-Aaiun', 'Yemen': 'Sanaa', 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare'}

# ==========================================================
# Regex patterns
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


LINKEDIN_REGEX = re.compile(
    r"(https?://)?(www\.)?linkedin\.com/[^\s]+",
    re.IGNORECASE,
)

GITHUB_REGEX = re.compile(
    r"(https?://)?(www\.)?github\.com/[^\s]+",
    re.IGNORECASE,
)

# ==========================================================
# Additional regex patterns
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


ADDRESS_KEYWORDS = {
    "address", "location", "home", "residence",
    "morada", "endereço", "based", "living"
}


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
    (?:-|–|—|\bto\b|\buntil\b|\btill\b|\bthrough\b|→|~)
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


EDU_KEYWORDS = {"education", "university", "school", "bachelor", "master","degree", "course", "student", "academic"}

WORK_KEYWORDS = {"experience", "work", "employment", "career", "history", "intern", "job", "company"}



# ==========================================================
# Text loading (Memory-based)
# ==========================================================
def clean_text(text: str) -> str:
    """Remove common OCR artifacts and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t\uf0b7\u2022*]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def normalize_text_input(text_input: Union[str, Dict, List]) -> str:
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

def normalize_phone(phone: str) -> Optional[str]:
    """
    Normalize phone numbers to international compact format.
    Example:
        '+351 968 736 666' -> '+351968736666'
    """
    digits = re.sub(r"[^\d+]", "", phone)

    # garantir tamanho mínimo plausível
    if len(re.sub(r"\D", "", digits)) < 9:
        return None

    # normalizar 00 → +
    if digits.startswith("00"):
        digits = "+" + digits[2:]

    # se não tiver código país, assume Portugal (+351)
    if not digits.startswith("+"):
        digits = "+351" + digits[-9:]

    return digits

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
    for country, capital in CAPITALS.items():
        if not capital:
            continue

        if re.search(rf"\b{re.escape(capital.lower())}\b", text_lower):
            return f"{capital}, {country}"

    # ---------- 2) Detect country ----------
    for country in CAPITALS.keys():
        if country.lower() in text_lower:
            return country

    return None



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



# ==========================================================
# Extraction helpers
# ==========================================================

def extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
    """Retorn the first ocorrence of a patern.."""
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def extract_all(pattern: re.Pattern, text: str) -> List[str]:
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

    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------
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
# Core extraction
# ==========================================================
class ContactInfo(TypedDict):
    email: Optional[str]
    all_emails: List[str]
    phone: Optional[str]
    all_phones: List[str]
    linkedin: Optional[str]
    github: Optional[str]
    address: Optional[str]
    experience_years: Optional[int]

def extract_contact_info(text: str) -> ContactInfo:

    """
    Extract info of the contacts and others of the text OCR of the CV.
    """

    emails = [normalize_email(e) for e in extract_all(EMAIL_REGEX, text)]
    raw_phones = extract_all(PHONE_REGEX, text)
    phones = []
    for p in raw_phones:
        norm = normalize_phone(p)
        if norm and norm not in phones:
            phones.append(norm)

    linkedin = extract_first(LINKEDIN_REGEX, text)
    github = extract_first(GITHUB_REGEX, text)
    address = extract_address(text)
    experience_years = extract_experience_years(text)

    return {
        "email": emails[0] if emails else None,
        "all_emails": emails,
        "phone": phones[0] if phones else None,
        "all_phones": phones,
        "linkedin": linkedin,
        "github": github,
        "address": address,
        "experience_years": experience_years,
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
    
    if isinstance(texts, dict) and not isinstance(texts.get("text"), str):
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
