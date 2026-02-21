import re
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional, Any, cast

from services.paths_models_and_data import get_gliner_model


# Define label sets globally for reuse
SKILL_LABELS = [
"skill", "technical expertise", "software and tools", "programming language",
    "software framework", "library", "database", "cloud platform", "devops tool",
    "operating system", "web technology", "machine learning algorithm",
    "data analysis tool", "big data tool", "version control system",
    "creative software", "project management tool", "office suite",
    "crm software", "erp software", "methodology", "development practice",
    "soft skill", "leadership skill", "personal attribute", "management skill",
    "competency", "interpersonal strength", "hard skill",
    "standard or protocol", "area of specialization"
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

UNIVERSITY_LABELS = [
    "university", "universidade", "institute",
    "school", "college", "faculty", "kth", "mit", "oxford", "harvard"]
UNIVERSITY_KEYWORDS =  {
    "university", "universidade", "institute",
    "school", "college", "faculty", "kth", "mit", "oxford", "harvard"}

DEGREE_LABELS = [
    "bachelor", "master", "phd", "doctorate", "doctor",
    "licenciatura", "mestrado", "doutoramento",
    "msc", "bsc", "mba",
    "engineering", "science", "arts"]

COMPANY_LABELS = [
    "ltd", "lda", "inc", "corp", "corporation",
    "technologies", "solutions", "group",
    "s.a.", "sa", "llc"]

ADDRESS_LABELS = [
    "street", "st.", "road", "rd.", "avenue", "ave",
    "rua", "avenida", "av.", "travessa",
    "city", "town", "country", "state", "region"]

OTHER_LABELS = [
    "person name",
    "job title",
    "company",
    "university",
    "degree",
    "address",]

ALL_LABELS = list(dict.fromkeys(SKILL_LABELS + OTHER_LABELS))

INVALID_ENTITIES = {
    "person name": {
        "person", "first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies"},
    "job title": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies"},
    "skills": {
        "person","first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "education", "experience", "hobbies", "C1", "First Certificate"},
    "degree": {"degree", "education"},
    "company": {"company", "organization"},
    "address": {"address", "location"}
        }

CAPITALS = {'Afghanistan': 'Kabul', 'Aland Islands': 'Mariehamn', 'Albania': 'Tirana', 'Algeria': 'Algiers', 'American Samoa': 'Pago Pago', 'Andorra': 'Andorra la Vella', 'Angola': 'Luanda', 'Anguilla': 'The Valley', 'Antarctica': '', 'Antigua And Barbuda': "St. John's", 'Argentina': 'Buenos Aires', 'Armenia': 'Yerevan', 'Aruba': 'Oranjestad', 'Australia': 'Canberra', 'Austria': 'Vienna', 'Azerbaijan': 'Baku', 'Bahamas The': 'Nassau', 'Bahrain': 'Manama', 'Bangladesh': 'Dhaka', 'Barbados': 'Bridgetown', 'Belarus': 'Minsk', 'Belgium': 'Brussels', 'Belize': 'Belmopan', 'Benin': 'Porto-Novo', 'Bermuda': 'Hamilton', 'Bhutan': 'Thimphu', 'Bolivia': 'Sucre', 'Bosnia and Herzegovina': 'Sarajevo', 'Botswana': 'Gaborone', 'Bouvet Island': '', 'Brazil': 'Brasilia', 'British Indian Ocean Territory': 'Diego Garcia', 'Brunei': 'Bandar Seri Begawan', 'Bulgaria': 'Sofia', 'Burkina Faso': 'Ouagadougou', 'Burundi': 'Bujumbura', 'Cambodia': 'Phnom Penh', 'Cameroon': 'Yaounde', 'Canada': 'Ottawa', 'Cape Verde': 'Praia', 'Cayman Islands': 'George Town', 'Central African Republic': 'Bangui', 'Chad': "N'Djamena", 'Chile': 'Santiago', 'China': 'Beijing', 'Christmas Island': 'Flying Fish Cove', 'Cocos (Keeling) Islands': 'West Island', 'Colombia': 'Bogota', 'Comoros': 'Moroni', 'Congo': 'Brazzaville', 'Congo The Democratic Republic Of The': 'Kinshasa', 'Cook Islands': 'Avarua', 'Costa Rica': 'San Jose', "Cote D'Ivoire (Ivory Coast)": 'Yamoussoukro', 'Croatia': 'Zagreb', 'Cuba': 'Havana', 'Cyprus': 'Nicosia', 'Czech Republic': 'Prague', 'Denmark': 'Copenhagen', 'Djibouti': 'Djibouti', 'Dominica': 'Roseau', 'Dominican Republic': 'Santo Domingo', 'East Timor': 'Dili', 'Ecuador': 'Quito', 'Egypt': 'Cairo', 'El Salvador': 'San Salvador', 'Equatorial Guinea': 'Malabo', 'Eritrea': 'Asmara', 'Estonia': 'Tallinn', 'Ethiopia': 'Addis Ababa', 'Falkland Islands': 'Stanley', 'Faroe Islands': 'Torshavn', 'Fiji Islands': 'Suva', 'Finland': 'Helsinki', 'France': 'Paris', 'French Guiana': 'Cayenne', 'French Polynesia': 'Papeete', 'French Southern Territories': 'Port-aux-Francais', 'Gabon': 'Libreville', 'Gambia The': 'Banjul', 'Georgia': 'Tbilisi', 'Germany': 'Berlin', 'Ghana': 'Accra', 'Gibraltar': 'Gibraltar', 'Greece': 'Athens', 'Greenland': 'Nuuk', 'Grenada': "St. George's", 'Guadeloupe': 'Basse-Terre', 'Guam': 'Hagatna', 'Guatemala': 'Guatemala City', 'Guernsey and Alderney': 'St Peter Port', 'Guinea': 'Conakry', 'Guinea-Bissau': 'Bissau', 'Guyana': 'Georgetown', 'Haiti': 'Port-au-Prince', 'Heard and McDonald Islands': '', 'Honduras': 'Tegucigalpa', 'Hong Kong S.A.R.': 'Hong Kong', 'Hungary': 'Budapest', 'Iceland': 'Reykjavik', 'India': 'New Delhi', 'Indonesia': 'Jakarta', 'Iran': 'Tehran', 'Iraq': 'Baghdad', 'Ireland': 'Dublin', 'Israel': 'Jerusalem', 'Italy': 'Rome', 'Jamaica': 'Kingston', 'Japan': 'Tokyo', 'Jersey': 'Saint Helier', 'Jordan': 'Amman', 'Kazakhstan': 'Astana', 'Kenya': 'Nairobi', 'Kiribati': 'Tarawa', 'Korea North': 'Pyongyang', 'Korea South': 'Seoul', 'Kuwait': 'Kuwait City', 'Kyrgyzstan': 'Bishkek', 'Laos': 'Vientiane', 'Latvia': 'Riga', 'Lebanon': 'Beirut', 'Lesotho': 'Maseru', 'Liberia': 'Monrovia', 'Libya': 'Tripolis', 'Liechtenstein': 'Vaduz', 'Lithuania': 'Vilnius', 'Luxembourg': 'Luxembourg', 'Macau S.A.R.': 'Macao', 'Macedonia': 'Skopje', 'Madagascar': 'Antananarivo', 'Malawi': 'Lilongwe', 'Malaysia': 'Kuala Lumpur', 'Maldives': 'Male', 'Mali': 'Bamako', 'Malta': 'Valletta', 'Man (Isle of)': 'Douglas, Isle of Man', 'Marshall Islands': 'Majuro', 'Martinique': 'Fort-de-France', 'Mauritania': 'Nouakchott', 'Mauritius': 'Port Louis', 'Mayotte': 'Mamoudzou', 'Mexico': 'Mexico City', 'Micronesia': 'Palikir', 'Moldova': 'Chisinau', 'Monaco': 'Monaco', 'Mongolia': 'Ulan Bator', 'Montenegro': 'Podgorica', 'Montserrat': 'Plymouth', 'Morocco': 'Rabat', 'Mozambique': 'Maputo', 'Myanmar': 'Nay Pyi Taw', 'Namibia': 'Windhoek', 'Nauru': 'Yaren', 'Nepal': 'Kathmandu', 'Netherlands Antilles': '', 'Netherlands The': 'Amsterdam', 'New Caledonia': 'Noumea', 'New Zealand': 'Wellington', 'Nicaragua': 'Managua', 'Niger': 'Niamey', 'Nigeria': 'Abuja', 'Niue': 'Alofi', 'Norfolk Island': 'Kingston', 'Northern Mariana Islands': 'Saipan', 'Norway': 'Oslo', 'Oman': 'Muscat', 'Pakistan': 'Islamabad', 'Palau': 'Melekeok', 'Palestinian Territory Occupied': 'East Jerusalem', 'Panama': 'Panama City', 'Papua new Guinea': 'Port Moresby', 'Paraguay': 'Asuncion', 'Peru': 'Lima', 'Philippines': 'Manila', 'Pitcairn Island': 'Adamstown', 'Poland': 'Warsaw', 'Portugal': 'Lisbon', 'Puerto Rico': 'San Juan', 'Qatar': 'Doha', 'Reunion': 'Saint-Denis', 'Romania': 'Bucharest', 'Russia': 'Moscow', 'Rwanda': 'Kigali', 'Saint Helena': 'Jamestown', 'Saint Kitts And Nevis': 'Basseterre', 'Saint Lucia': 'Castries', 'Saint Pierre and Miquelon': 'Saint-Pierre', 'Saint Vincent And The Grenadines': 'Kingstown', 'Saint-Barthelemy': 'Gustavia', 'Saint-Martin (French part)': 'Marigot', 'Samoa': 'Apia', 'San Marino': 'San Marino', 'Sao Tome and Principe': 'Sao Tome', 'Saudi Arabia': 'Riyadh', 'Senegal': 'Dakar', 'Serbia': 'Belgrade', 'Seychelles': 'Victoria', 'Sierra Leone': 'Freetown', 'Singapore': 'Singapur', 'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Solomon Islands': 'Honiara', 'Somalia': 'Mogadishu', 'South Africa': 'Pretoria', 'South Georgia': 'Grytviken', 'South Sudan': 'Juba', 'Spain': 'Madrid', 'Sri Lanka': 'Colombo', 'Sudan': 'Khartoum', 'Suriname': 'Paramaribo', 'Svalbard And Jan Mayen Islands': 'Longyearbyen', 'Swaziland': 'Mbabane', 'Sweden': 'Stockholm', 'Switzerland': 'Berne', 'Syria': 'Damascus', 'Taiwan': 'Taipei', 'Tajikistan': 'Dushanbe', 'Tanzania': 'Dodoma', 'Thailand': 'Bangkok', 'Togo': 'Lome', 'Tokelau': '', 'Tonga': "Nuku'alofa", 'Trinidad And Tobago': 'Port of Spain', 'Tunisia': 'Tunis', 'Turkey': 'Ankara', 'Turkmenistan': 'Ashgabat', 'Turks And Caicos Islands': 'Cockburn Town', 'Tuvalu': 'Funafuti', 'Uganda': 'Kampala', 'Ukraine': 'Kiev', 'United Arab Emirates': 'Abu Dhabi', 'United Kingdom': 'London', 'United States': 'Washington', 'United States Minor Outlying Islands': '', 'Uruguay': 'Montevideo', 'Uzbekistan': 'Tashkent', 'Vanuatu': 'Port Vila', 'Vatican City State (Holy See)': 'Vatican City', 'Venezuela': 'Caracas', 'Vietnam': 'Hanoi', 'British Virgin Islands': 'Road Town', 'U.S. Virgin Islands': 'Charlotte Amalie', 'USA Virgin Islands': 'Charlotte Amalie', 'Wallis And Futuna Islands': 'Mata Utu', 'Western Sahara': 'El-Aaiun', 'Yemen': 'Sanaa', 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare'}
COUNTRIES = set(CAPITALS.keys())
CAPITAL_NAMES = {v for v in CAPITALS.values() if v}
POSTAL_PATTERN = re.compile(r"\b\d{4,6}[- ]?\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\b(nº|no\.?|number)?\s?\d+\b", re.IGNORECASE)

NAME_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")



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

def _clean_degree_entities(values: List[str]) -> List[str]:
    """
    Keep full degree descriptions.
    Remove exact duplicates only.
    """
    return list(dict.fromkeys(values))

def _expand_address_smart(full_text: str, address_fragment: str, window: int = 60) -> str:
    """
    Expand address fragment by searching for structural address signals
    (postal codes, numbers, capitals, countries) within a character window.
    """

    match = re.search(re.escape(address_fragment), full_text)
    if not match:
        return address_fragment

    start = match.start()
    end = match.end()

    # Define search boundaries
    left_bound = max(0, start - window)
    right_bound = min(len(full_text), end + window)

    segment = full_text[left_bound:right_bound]

    # Split into tokens with positions
    expanded_start = start
    expanded_end = end

    # Search right side first
    right_text = full_text[end:right_bound]

    for pattern in [POSTAL_PATTERN, NUMBER_PATTERN]:
        m = pattern.search(right_text)
        if m:
            expanded_end = end + m.end()

    for name in CAPITAL_NAMES.union(COUNTRIES):
        idx = right_text.lower().find(name.lower())
        if idx != -1:
            expanded_end = end + idx + len(name)

    # Then search left side
    left_text = full_text[left_bound:start]

    for pattern in [NUMBER_PATTERN]:
        matches = list(pattern.finditer(left_text))
        if matches:
            expanded_start = left_bound + matches[-1].start()

    expanded = full_text[expanded_start:expanded_end]

    # Clean formatting
    expanded = re.sub(r"\s+", " ", expanded).strip(" ,")

    return expanded

def extract_entities_from_text(
    text: str,
    model: Optional[GLiNER] = None,
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
    if not text or not text.strip():
        result_labels = ["person name","job title","company","university","degree","address","skills"]
        return {label: [] for label in result_labels}

    # Load default model only if not provided
    if model is None:
        model = get_gliner_model()

    model = cast(GLiNER, model)

    # Predict entities (robust to model/runtime errors)
    try:
        entities = model.predict_entities(text, ALL_LABELS, threshold=threshold) # type: ignore[attr-defined]
    except Exception:
        # Return empty structure instead of crashing the whole pipeline
        result_labels = ["person name","job title","company","university","degree","address","skills"]
        return {label: [] for label in result_labels}

    
    # Organize entities by label
    result_labels = ["person name","job title","company","university","degree","address","skills"]
    entities_by_label = {label: [] for label in result_labels}
    
    for entity in entities:
        raw_text = entity["text"]
        text_norm = raw_text.lower().strip()
        label = entity["label"]

        # ----------------------------------------------
        # Remove invalid generic tokens
        # ----------------------------------------------
        invalid_set = INVALID_ENTITIES.get(label, set())
        if text_norm in invalid_set or len(text_norm) <= 2:
            continue

        # ----------------------------------------------
        # Skill logic (priority)
        # ----------------------------------------------
        if label in SKILL_LABELS:
            entities_by_label["skills"].append(raw_text)
            continue

        if any(re.search(rf"\b{re.escape(sw)}\b", text_norm) for sw in SOFTWARE_AS_SKILL):
            entities_by_label["skills"].append(raw_text)
            continue

        # ----------------------------------------------
        # Degree validation (positive validation)
        # ----------------------------------------------
        if label == "degree":
            entities_by_label["degree"].append(raw_text)
            continue

        # ----------------------------------------------
        # Company vs University classification
        # ----------------------------------------------
        if label == "company":
            entities_by_label["company"].append(raw_text)
            continue

        # ----------------------------------------------
        # Address validation (positive)
        # ----------------------------------------------
        if label == "address":
            expanded = _expand_address_smart(text, raw_text)
            entities_by_label["address"].append(expanded)
            continue

        # ----------------------------------------------
        # Person name & job title
        # ----------------------------------------------
        if label in {"person name", "job title"}:
            entities_by_label[label].append(raw_text)
    
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
        DataFrame with columns: skills, person_name, job_title, company, 
                               university, degree, address
    
    Example:
        >>> from text_GliNER_preprocessing import preprocess_ocr_for_gliner
        >>> texts = [preprocess_ocr_for_gliner("Email: john@example.com\nSkills: Python, Java"), ...]
        >>> df = process_texts_gliner(texts)
    """
    if model is None:
        model = get_gliner_model()


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
            row: Dict[str, Any] = {}
            result_labels = ["person name","job title","company","university","degree","address","skills"]
            for label in result_labels:
                entities = entities_dict.get(label, [])

                # --- limpeza robusta ---
                clean_entities = []
                for e in entities:
                    if not isinstance(e, str):
                        continue

                    e_clean = " ".join(e.replace("\n", " ").replace("\r", " ").split())

                    if e_clean:
                        clean_entities.append(e_clean.strip())

                # -----------------------------------------
                # LABEL-SPECIFIC POST-PROCESSING
                # -----------------------------------------

                if label == "person name":
                    clean_entities = _clean_person_name(clean_entities)

                if label in {"university", "company", "address"}:
                    clean_entities = _remove_substring_duplicates(clean_entities)

                if label == "degree":
                    clean_entities = _clean_degree_entities(clean_entities)

                

                # Remove duplicates ignoring case
                unique = {}
                for e in clean_entities:
                    key = e.lower()
                    if key not in unique:
                        unique[key] = e

                clean_entities = list(unique.values())

                if label == "company":
                    uni = []
                    comp = []
                    for e in clean_entities:
                        e_low = e.lower()
                        if any(k in e_low for k in UNIVERSITY_KEYWORDS):
                            uni.append(e)
                        else:
                            comp.append(e)

                    # guarda university e mantém company separado
                    row["university"] = ", ".join(_remove_substring_duplicates(uni)) if uni else None
                    clean_entities = comp

                row[label] = ", ".join(clean_entities) if clean_entities else None

                            
            rows.append(row)
        
        except Exception as e:
            rows.append({"error": str(e)})
    
    df = pd.DataFrame(rows)
    return df

