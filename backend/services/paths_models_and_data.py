"""
Centralized paths and model loaders.

This module:
• Defines all backend paths
• Lazily loads heavy ML models (GLiNER, YOLO)
• Guarantees single instance per process (thread-safe)
• Prevents repeated disk/model loading
"""

from pathlib import Path
from typing import Optional, Any
from threading import Lock

from gliner import GLiNER
from ultralytics import YOLO  # type: ignore
import pytesseract

# =========================================================
# PATHS
# =========================================================

# Projects root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"

# Data folders
UPLOADS_DIR = BACKEND_ROOT / "uploads"
OUTPUTS_DIR = BACKEND_ROOT / "outputs"

# Model paths
GLINER_MODEL_PATH = BACKEND_ROOT / "models" / "gliner"
FILTER_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11n-text.pt"
TEXT_PHOTO_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11s-text-photo-150e4b.pt"
SKILL_TABLE_YOLO_MODEL_PATH = BACKEND_ROOT / "models" / "yolo" / "yolo11s-skill_table-150e4b.pt"


# =========================================================
# DATA
# =========================================================
LANGUAGES = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Aymara', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bislama', 'Bosnian', 'Bulgarian', 'Burmese', 'Castilian', 'Catalan', 'Cantonese','Chamorro', 'Chichewa', 'Chinese', 'Chinese (Simplified)','Croatian', 'Czech', 'Danish', 'Divehi', 'Dutch', 'Dzongkha', 'English', 'Estonian', 'Faroese', 'Fijian', 'Finnish', 'French', 'Fulah', 'Galician', 'Georgian', 'German', 'Guarani', 'Haitian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Kalaallisut', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Kirghiz', 'Kongo', 'Korean', 'Kurdish', 'Lao', 'Latin', 'Latvian', 'Lingala', 'Lithuanian', 'Luba-Katanga', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Maltese', 'Mandarin','Manx', 'Maori', 'Marshallese', 'Modern Greek', 'Greek', 'Mongolian', 'Nauru', 'Nepali', 'North Ndebele', 'Norwegian', 'Norwegian Bokmål', 'Norwegian Nynorsk', 'Occitan (post 1500)', 'Panjabi', 'Persian', 'Polish', 'Portuguese', 'Pushto', 'Quechua', 'Romanian', 'Rundi', 'Russian', 'Samoan', 'Sango', 'Serbian', 'Shona', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'South Ndebele', 'Southern Sotho', 'Spanish', 'Swahili', 'Swati', 'Swedish', 'Tajik', 'Tamil', 'Thai', 'Tigrinya', 'Tonga (Tonga Islands)', 'Tsonga', 'Tswana', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Xhosa', 'Zulu']
LANGUAGES_SMALL=['af', 'am', 'ar', 'ay', 'az', 'be', 'bg', 'bi', 'bn', 'bs', 'ca', 'ch', 'cs', 'da', 'de', 'dv', 'dz', 'el', 'en', 'es', 'et', 'eu', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'ga', 'gl', 'gn', 'gv', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kg', 'kk', 'kl', 'km', 'ko', 'ku', 'ky', 'la', 'lb', 'ln', 'lo', 'lt', 'lu', 'lv', 'mg', 'mh', 'mi', 'mk', 'mn', 'ms', 'mt', 'my', 'na', 'nb', 'nd', 'ne', 'nl', 'nn', 'no', 'nr', 'ny', 'oc', 'pa', 'pl', 'ps', 'pt', 'qu', 'rn', 'ro', 'ru', 'rw', 'sg', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'sv', 'sw', 'ta', 'tg', 'th', 'ti', 'tk', 'tn', 'to', 'tr', 'ts', 'uk', 'ur', 'uz', 've', 'vi', 'xh', 'zh', 'zu']

COUNTRIES_AND_CAPITALS = {'Afghanistan': 'Kabul', 'Aland Islands': 'Mariehamn', 'Albania': 'Tirana', 'Algeria': 'Algiers', 'American Samoa': 'Pago Pago', 'Andorra': 'Andorra la Vella', 'Angola': 'Luanda', 'Anguilla': 'The Valley', 'Antarctica': '', 'Antigua And Barbuda': "St. John's", 'Argentina': 'Buenos Aires', 'Armenia': 'Yerevan', 'Aruba': 'Oranjestad', 'Australia': 'Canberra', 'Austria': 'Vienna', 'Azerbaijan': 'Baku', 'Bahamas The': 'Nassau', 'Bahrain': 'Manama', 'Bangladesh': 'Dhaka', 'Barbados': 'Bridgetown', 'Belarus': 'Minsk', 'Belgium': 'Brussels', 'Belize': 'Belmopan', 'Benin': 'Porto-Novo', 'Bermuda': 'Hamilton', 'Bhutan': 'Thimphu', 'Bolivia': 'Sucre', 'Bosnia and Herzegovina': 'Sarajevo', 'Botswana': 'Gaborone', 'Bouvet Island': '', 'Brazil': 'Brasilia', 'British Indian Ocean Territory': 'Diego Garcia', 'Brunei': 'Bandar Seri Begawan', 'Bulgaria': 'Sofia', 'Burkina Faso': 'Ouagadougou', 'Burundi': 'Bujumbura', 'Cambodia': 'Phnom Penh', 'Cameroon': 'Yaounde', 'Canada': 'Ottawa', 'Cape Verde': 'Praia', 'Cayman Islands': 'George Town', 'Central African Republic': 'Bangui', 'Chad': "N'Djamena", 'Chile': 'Santiago', 'China': 'Beijing', 'Christmas Island': 'Flying Fish Cove', 'Cocos (Keeling) Islands': 'West Island', 'Colombia': 'Bogota', 'Comoros': 'Moroni', 'Congo': 'Brazzaville', 'Congo The Democratic Republic Of The': 'Kinshasa', 'Cook Islands': 'Avarua', 'Costa Rica': 'San Jose', "Cote D'Ivoire (Ivory Coast)": 'Yamoussoukro', 'Croatia': 'Zagreb', 'Cuba': 'Havana', 'Cyprus': 'Nicosia', 'Czech Republic': 'Prague', 'Denmark': 'Copenhagen', 'Djibouti': 'Djibouti', 'Dominica': 'Roseau', 'Dominican Republic': 'Santo Domingo', 'East Timor': 'Dili', 'Ecuador': 'Quito', 'Egypt': 'Cairo', 'El Salvador': 'San Salvador', 'Equatorial Guinea': 'Malabo', 'Eritrea': 'Asmara', 'Estonia': 'Tallinn', 'Ethiopia': 'Addis Ababa', 'Falkland Islands': 'Stanley', 'Faroe Islands': 'Torshavn', 'Fiji Islands': 'Suva', 'Finland': 'Helsinki', 'France': 'Paris', 'French Guiana': 'Cayenne', 'French Polynesia': 'Papeete', 'French Southern Territories': 'Port-aux-Francais', 'Gabon': 'Libreville', 'Gambia The': 'Banjul', 'Georgia': 'Tbilisi', 'Germany': 'Berlin', 'Ghana': 'Accra', 'Gibraltar': 'Gibraltar', 'Greece': 'Athens', 'Greenland': 'Nuuk', 'Grenada': "St. George's", 'Guadeloupe': 'Basse-Terre', 'Guam': 'Hagatna', 'Guatemala': 'Guatemala City', 'Guernsey and Alderney': 'St Peter Port', 'Guinea': 'Conakry', 'Guinea-Bissau': 'Bissau', 'Guyana': 'Georgetown', 'Haiti': 'Port-au-Prince', 'Heard and McDonald Islands': '', 'Honduras': 'Tegucigalpa', 'Hong Kong S.A.R.': 'Hong Kong', 'Hungary': 'Budapest', 'Iceland': 'Reykjavik', 'India': 'New Delhi', 'Indonesia': 'Jakarta', 'Iran': 'Tehran', 'Iraq': 'Baghdad', 'Ireland': 'Dublin', 'Israel': 'Jerusalem', 'Italy': 'Rome', 'Jamaica': 'Kingston', 'Japan': 'Tokyo', 'Jersey': 'Saint Helier', 'Jordan': 'Amman', 'Kazakhstan': 'Astana', 'Kenya': 'Nairobi', 'Kiribati': 'Tarawa', 'Korea North': 'Pyongyang', 'Korea South': 'Seoul', 'Kuwait': 'Kuwait City', 'Kyrgyzstan': 'Bishkek', 'Laos': 'Vientiane', 'Latvia': 'Riga', 'Lebanon': 'Beirut', 'Lesotho': 'Maseru', 'Liberia': 'Monrovia', 'Libya': 'Tripolis', 'Liechtenstein': 'Vaduz', 'Lithuania': 'Vilnius', 'Luxembourg': 'Luxembourg', 'Macau S.A.R.': 'Macao', 'Macedonia': 'Skopje', 'Madagascar': 'Antananarivo', 'Malawi': 'Lilongwe', 'Malaysia': 'Kuala Lumpur', 'Maldives': 'Male', 'Mali': 'Bamako', 'Malta': 'Valletta', 'Man (Isle of)': 'Douglas, Isle of Man', 'Marshall Islands': 'Majuro', 'Martinique': 'Fort-de-France', 'Mauritania': 'Nouakchott', 'Mauritius': 'Port Louis', 'Mayotte': 'Mamoudzou', 'Mexico': 'Mexico City', 'Micronesia': 'Palikir', 'Moldova': 'Chisinau', 'Monaco': 'Monaco', 'Mongolia': 'Ulan Bator', 'Montenegro': 'Podgorica', 'Montserrat': 'Plymouth', 'Morocco': 'Rabat', 'Mozambique': 'Maputo', 'Myanmar': 'Nay Pyi Taw', 'Namibia': 'Windhoek', 'Nauru': 'Yaren', 'Nepal': 'Kathmandu', 'Netherlands Antilles': '', 'Netherlands The': 'Amsterdam', 'New Caledonia': 'Noumea', 'New Zealand': 'Wellington', 'Nicaragua': 'Managua', 'Niger': 'Niamey', 'Nigeria': 'Abuja', 'Niue': 'Alofi', 'Norfolk Island': 'Kingston', 'Northern Mariana Islands': 'Saipan', 'Norway': 'Oslo', 'Oman': 'Muscat', 'Pakistan': 'Islamabad', 'Palau': 'Melekeok', 'Palestinian Territory Occupied': 'East Jerusalem', 'Panama': 'Panama City', 'Papua new Guinea': 'Port Moresby', 'Paraguay': 'Asuncion', 'Peru': 'Lima', 'Philippines': 'Manila', 'Pitcairn Island': 'Adamstown', 'Poland': 'Warsaw', 'Portugal': 'Lisbon', 'Puerto Rico': 'San Juan', 'Qatar': 'Doha', 'Reunion': 'Saint-Denis', 'Romania': 'Bucharest', 'Russia': 'Moscow', 'Rwanda': 'Kigali', 'Saint Helena': 'Jamestown', 'Saint Kitts And Nevis': 'Basseterre', 'Saint Lucia': 'Castries', 'Saint Pierre and Miquelon': 'Saint-Pierre', 'Saint Vincent And The Grenadines': 'Kingstown', 'Saint-Barthelemy': 'Gustavia', 'Saint-Martin (French part)': 'Marigot', 'Samoa': 'Apia', 'San Marino': 'San Marino', 'Sao Tome and Principe': 'Sao Tome', 'Saudi Arabia': 'Riyadh', 'Senegal': 'Dakar', 'Serbia': 'Belgrade', 'Seychelles': 'Victoria', 'Sierra Leone': 'Freetown', 'Singapore': 'Singapur', 'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Solomon Islands': 'Honiara', 'Somalia': 'Mogadishu', 'South Africa': 'Pretoria', 'South Georgia': 'Grytviken', 'South Sudan': 'Juba', 'Spain': 'Madrid', 'Sri Lanka': 'Colombo', 'Sudan': 'Khartoum', 'Suriname': 'Paramaribo', 'Svalbard And Jan Mayen Islands': 'Longyearbyen', 'Swaziland': 'Mbabane', 'Sweden': 'Stockholm', 'Switzerland': 'Berne', 'Syria': 'Damascus', 'Taiwan': 'Taipei', 'Tajikistan': 'Dushanbe', 'Tanzania': 'Dodoma', 'Thailand': 'Bangkok', 'Togo': 'Lome', 'Tokelau': '', 'Tonga': "Nuku'alofa", 'Trinidad And Tobago': 'Port of Spain', 'Tunisia': 'Tunis', 'Turkey': 'Ankara', 'Turkmenistan': 'Ashgabat', 'Turks And Caicos Islands': 'Cockburn Town', 'Tuvalu': 'Funafuti', 'Uganda': 'Kampala', 'Ukraine': 'Kiev', 'United Arab Emirates': 'Abu Dhabi', 'United Kingdom': 'London', 'United States': 'Washington', 'United States Minor Outlying Islands': '', 'Uruguay': 'Montevideo', 'Uzbekistan': 'Tashkent', 'Vanuatu': 'Port Vila', 'Vatican City State (Holy See)': 'Vatican City', 'Venezuela': 'Caracas', 'Vietnam': 'Hanoi', 'British Virgin Islands': 'Road Town', 'U.S. Virgin Islands': 'Charlotte Amalie', 'USA Virgin Islands': 'Charlotte Amalie', 'Wallis And Futuna Islands': 'Mata Utu', 'Western Sahara': 'El-Aaiun', 'Yemen': 'Sanaa', 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare'}

COUNTRIES_NAMES = set(COUNTRIES_AND_CAPITALS .keys())
CAPITALS_NAMES = {v for v in COUNTRIES_AND_CAPITALS .values() if v}

CITIES_USA = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Jacksonville", "Fort Worth", "San Jose", "Austin", "Charlotte", "Columbus","Indianapolis", "San Francisco", "Seattle", "Denver", "Oklahoma City", "Nashville", "El Paso", "Washington D.C.", "Las Vegas", "Boston", "Portland", "Louisville", "Memphis", "Detroit", "Albuquerque", "Baltimore", "Milwaukee", "Fresno", "Sacramento", "Mesa", "Kansas City", "Atlanta", "Colorado Springs", "Raleigh", "Omaha", "Miami", "Virginia Beach", "Long Beach", "Oakland", "Minneapolis", "Bakersfield", "Tulsa", "Tampa", "Aurora"]
CITIES_PORTUGAL = ['Porto', 'Setubal', "Setúbal", "Castelo Branco"]


SKILL_LABELS = [
"skill", "technical expertise", "software and tools", "programming language",
    "software framework", "library", "database", "cloud platform", "devops tool",
    "operating system", "web technology", "machine learning algorithm",
    "data analysis tool", "big data tool", "version control system",
    "creative software", "project management tool", "office suite",
    "crm software", "erp software", "methodology", "development practice",
    "soft skill", "leadership skill", "management skill",
    "competency", "interpersonal strength", "hard skill", "artistic skill",
    "physical skills", "mental skills","standard or protocol", "area of specialization"]

# =========================================================
# GLOBAL MODEL CACHE (thread-safe singletons)
# =========================================================

_GLINER_MODEL: Optional[Any] = None
_FILTER_YOLO_MODEL: Optional[YOLO] = None
_TEXT_PHOTO_YOLO_MODEL: Optional[YOLO] = None
_SKILL_TABLE_YOLO_MODEL: Optional[YOLO] = None

_GLINER_LOCK = Lock()
_FILTER_LOCK = Lock()
_TEXT_PHOTO_LOCK = Lock()
_SKILL_TABLE_LOCK = Lock()


# =========================================================
# GLiNER
# =========================================================

def get_gliner_model() -> GLiNER:
    """Return cached GLiNER model (loaded once)."""
    global _GLINER_MODEL

    if _GLINER_MODEL is None:
        with _GLINER_LOCK:
            if _GLINER_MODEL is None:
                if not GLINER_MODEL_PATH.exists():
                    raise FileNotFoundError(f"GLiNER model not found: {GLINER_MODEL_PATH}")

                _GLINER_MODEL = GLiNER.from_pretrained(str(GLINER_MODEL_PATH))

    return _GLINER_MODEL


# =========================================================
# YOLO — CV FILTER
# =========================================================

def get_filter_yolo_model() -> YOLO:
    """Return cached YOLO model for CV detection."""
    global _FILTER_YOLO_MODEL

    if _FILTER_YOLO_MODEL is None:
        with _FILTER_LOCK:
            if _FILTER_YOLO_MODEL is None:
                if not FILTER_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO filter model not found: {FILTER_YOLO_MODEL_PATH}")

                _FILTER_YOLO_MODEL = YOLO(str(FILTER_YOLO_MODEL_PATH))

    return _FILTER_YOLO_MODEL


# =========================================================
# YOLO — TEXT / PHOTO
# =========================================================

def get_text_photo_yolo_model() -> YOLO:
    """Return cached YOLO model for text/photo detection."""
    global _TEXT_PHOTO_YOLO_MODEL

    if _TEXT_PHOTO_YOLO_MODEL is None:
        with _TEXT_PHOTO_LOCK:
            if _TEXT_PHOTO_YOLO_MODEL is None:
                if not TEXT_PHOTO_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO text/photo model not found: {TEXT_PHOTO_YOLO_MODEL_PATH}")

                _TEXT_PHOTO_YOLO_MODEL = YOLO(str(TEXT_PHOTO_YOLO_MODEL_PATH))

    return _TEXT_PHOTO_YOLO_MODEL


# =========================================================
# YOLO — SKILL TABLE
# =========================================================

def get_skill_table_yolo_model() -> YOLO:
    """Return cached YOLO model for skill table detection."""
    global _SKILL_TABLE_YOLO_MODEL

    if _SKILL_TABLE_YOLO_MODEL is None:
        with _SKILL_TABLE_LOCK:
            if _SKILL_TABLE_YOLO_MODEL is None:
                if not SKILL_TABLE_YOLO_MODEL_PATH.exists():
                    raise FileNotFoundError(f"YOLO skill table model not found: {SKILL_TABLE_YOLO_MODEL_PATH}")

                _SKILL_TABLE_YOLO_MODEL = YOLO(str(SKILL_TABLE_YOLO_MODEL_PATH))

    return _SKILL_TABLE_YOLO_MODEL


# ---------------------------------------------------------
# TESSERACT CONFIGURATION
# ---------------------------------------------------------

# Default Windows installation path, if you have custom chose custom
TESSERACT_EXE_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

_TESSERACT_CONFIGURED = False
_TESSERACT_LOCK = Lock()

def ensure_tesseract_configured() -> None:
    """
    Configure Tesseract executable path once per process.

    Unlike YOLO or GLiNER, Tesseract is NOT a Python model in memory.
    It is an external executable called on each OCR request.

    This function only ensures the executable path is set.
    """
    global _TESSERACT_CONFIGURED

    if not _TESSERACT_CONFIGURED:
        with _TESSERACT_LOCK:
            if not _TESSERACT_CONFIGURED:

                if not TESSERACT_EXE_PATH.exists():
                    raise FileNotFoundError(
                        f"Tesseract executable not found at: {TESSERACT_EXE_PATH}\n"
                        "Install Tesseract OCR or update TESSERACT_EXE_PATH."
                    )

                pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE_PATH)
                _TESSERACT_CONFIGURED = True

def run_tesseract_ocr(image, lang: str = "eng", config: str = "--oem 3 --psm 6") -> str:
    """
    Execute OCR using the configured Tesseract engine.

    Parameters
    ----------
    image : PIL.Image | numpy.ndarray
    lang : str
        OCR language code.
    config : str
        Tesseract configuration string.

    Returns
    -------
    str
        Extracted text.
    """
    ensure_tesseract_configured()

    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=config,
    )

    return text.strip()
