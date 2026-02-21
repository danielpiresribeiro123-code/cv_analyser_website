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
    "competency", "interpersonal strength", "hard skill", "artistic skill",
    "physical skills", "mental skills","standard or protocol", "area of specialization"]

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

COUNTRIES_AND_CAPITALS = {'Afghanistan': 'Kabul', 'Aland Islands': 'Mariehamn', 'Albania': 'Tirana', 'Algeria': 'Algiers', 'American Samoa': 'Pago Pago', 'Andorra': 'Andorra la Vella', 'Angola': 'Luanda', 'Anguilla': 'The Valley', 'Antarctica': '', 'Antigua And Barbuda': "St. John's", 'Argentina': 'Buenos Aires', 'Armenia': 'Yerevan', 'Aruba': 'Oranjestad', 'Australia': 'Canberra', 'Austria': 'Vienna', 'Azerbaijan': 'Baku', 'Bahamas The': 'Nassau', 'Bahrain': 'Manama', 'Bangladesh': 'Dhaka', 'Barbados': 'Bridgetown', 'Belarus': 'Minsk', 'Belgium': 'Brussels', 'Belize': 'Belmopan', 'Benin': 'Porto-Novo', 'Bermuda': 'Hamilton', 'Bhutan': 'Thimphu', 'Bolivia': 'Sucre', 'Bosnia and Herzegovina': 'Sarajevo', 'Botswana': 'Gaborone', 'Bouvet Island': '', 'Brazil': 'Brasilia', 'British Indian Ocean Territory': 'Diego Garcia', 'Brunei': 'Bandar Seri Begawan', 'Bulgaria': 'Sofia', 'Burkina Faso': 'Ouagadougou', 'Burundi': 'Bujumbura', 'Cambodia': 'Phnom Penh', 'Cameroon': 'Yaounde', 'Canada': 'Ottawa', 'Cape Verde': 'Praia', 'Cayman Islands': 'George Town', 'Central African Republic': 'Bangui', 'Chad': "N'Djamena", 'Chile': 'Santiago', 'China': 'Beijing', 'Christmas Island': 'Flying Fish Cove', 'Cocos (Keeling) Islands': 'West Island', 'Colombia': 'Bogota', 'Comoros': 'Moroni', 'Congo': 'Brazzaville', 'Congo The Democratic Republic Of The': 'Kinshasa', 'Cook Islands': 'Avarua', 'Costa Rica': 'San Jose', "Cote D'Ivoire (Ivory Coast)": 'Yamoussoukro', 'Croatia': 'Zagreb', 'Cuba': 'Havana', 'Cyprus': 'Nicosia', 'Czech Republic': 'Prague', 'Denmark': 'Copenhagen', 'Djibouti': 'Djibouti', 'Dominica': 'Roseau', 'Dominican Republic': 'Santo Domingo', 'East Timor': 'Dili', 'Ecuador': 'Quito', 'Egypt': 'Cairo', 'El Salvador': 'San Salvador', 'Equatorial Guinea': 'Malabo', 'Eritrea': 'Asmara', 'Estonia': 'Tallinn', 'Ethiopia': 'Addis Ababa', 'Falkland Islands': 'Stanley', 'Faroe Islands': 'Torshavn', 'Fiji Islands': 'Suva', 'Finland': 'Helsinki', 'France': 'Paris', 'French Guiana': 'Cayenne', 'French Polynesia': 'Papeete', 'French Southern Territories': 'Port-aux-Francais', 'Gabon': 'Libreville', 'Gambia The': 'Banjul', 'Georgia': 'Tbilisi', 'Germany': 'Berlin', 'Ghana': 'Accra', 'Gibraltar': 'Gibraltar', 'Greece': 'Athens', 'Greenland': 'Nuuk', 'Grenada': "St. George's", 'Guadeloupe': 'Basse-Terre', 'Guam': 'Hagatna', 'Guatemala': 'Guatemala City', 'Guernsey and Alderney': 'St Peter Port', 'Guinea': 'Conakry', 'Guinea-Bissau': 'Bissau', 'Guyana': 'Georgetown', 'Haiti': 'Port-au-Prince', 'Heard and McDonald Islands': '', 'Honduras': 'Tegucigalpa', 'Hong Kong S.A.R.': 'Hong Kong', 'Hungary': 'Budapest', 'Iceland': 'Reykjavik', 'India': 'New Delhi', 'Indonesia': 'Jakarta', 'Iran': 'Tehran', 'Iraq': 'Baghdad', 'Ireland': 'Dublin', 'Israel': 'Jerusalem', 'Italy': 'Rome', 'Jamaica': 'Kingston', 'Japan': 'Tokyo', 'Jersey': 'Saint Helier', 'Jordan': 'Amman', 'Kazakhstan': 'Astana', 'Kenya': 'Nairobi', 'Kiribati': 'Tarawa', 'Korea North': 'Pyongyang', 'Korea South': 'Seoul', 'Kuwait': 'Kuwait City', 'Kyrgyzstan': 'Bishkek', 'Laos': 'Vientiane', 'Latvia': 'Riga', 'Lebanon': 'Beirut', 'Lesotho': 'Maseru', 'Liberia': 'Monrovia', 'Libya': 'Tripolis', 'Liechtenstein': 'Vaduz', 'Lithuania': 'Vilnius', 'Luxembourg': 'Luxembourg', 'Macau S.A.R.': 'Macao', 'Macedonia': 'Skopje', 'Madagascar': 'Antananarivo', 'Malawi': 'Lilongwe', 'Malaysia': 'Kuala Lumpur', 'Maldives': 'Male', 'Mali': 'Bamako', 'Malta': 'Valletta', 'Man (Isle of)': 'Douglas, Isle of Man', 'Marshall Islands': 'Majuro', 'Martinique': 'Fort-de-France', 'Mauritania': 'Nouakchott', 'Mauritius': 'Port Louis', 'Mayotte': 'Mamoudzou', 'Mexico': 'Mexico City', 'Micronesia': 'Palikir', 'Moldova': 'Chisinau', 'Monaco': 'Monaco', 'Mongolia': 'Ulan Bator', 'Montenegro': 'Podgorica', 'Montserrat': 'Plymouth', 'Morocco': 'Rabat', 'Mozambique': 'Maputo', 'Myanmar': 'Nay Pyi Taw', 'Namibia': 'Windhoek', 'Nauru': 'Yaren', 'Nepal': 'Kathmandu', 'Netherlands Antilles': '', 'Netherlands The': 'Amsterdam', 'New Caledonia': 'Noumea', 'New Zealand': 'Wellington', 'Nicaragua': 'Managua', 'Niger': 'Niamey', 'Nigeria': 'Abuja', 'Niue': 'Alofi', 'Norfolk Island': 'Kingston', 'Northern Mariana Islands': 'Saipan', 'Norway': 'Oslo', 'Oman': 'Muscat', 'Pakistan': 'Islamabad', 'Palau': 'Melekeok', 'Palestinian Territory Occupied': 'East Jerusalem', 'Panama': 'Panama City', 'Papua new Guinea': 'Port Moresby', 'Paraguay': 'Asuncion', 'Peru': 'Lima', 'Philippines': 'Manila', 'Pitcairn Island': 'Adamstown', 'Poland': 'Warsaw', 'Portugal': 'Lisbon', 'Puerto Rico': 'San Juan', 'Qatar': 'Doha', 'Reunion': 'Saint-Denis', 'Romania': 'Bucharest', 'Russia': 'Moscow', 'Rwanda': 'Kigali', 'Saint Helena': 'Jamestown', 'Saint Kitts And Nevis': 'Basseterre', 'Saint Lucia': 'Castries', 'Saint Pierre and Miquelon': 'Saint-Pierre', 'Saint Vincent And The Grenadines': 'Kingstown', 'Saint-Barthelemy': 'Gustavia', 'Saint-Martin (French part)': 'Marigot', 'Samoa': 'Apia', 'San Marino': 'San Marino', 'Sao Tome and Principe': 'Sao Tome', 'Saudi Arabia': 'Riyadh', 'Senegal': 'Dakar', 'Serbia': 'Belgrade', 'Seychelles': 'Victoria', 'Sierra Leone': 'Freetown', 'Singapore': 'Singapur', 'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Solomon Islands': 'Honiara', 'Somalia': 'Mogadishu', 'South Africa': 'Pretoria', 'South Georgia': 'Grytviken', 'South Sudan': 'Juba', 'Spain': 'Madrid', 'Sri Lanka': 'Colombo', 'Sudan': 'Khartoum', 'Suriname': 'Paramaribo', 'Svalbard And Jan Mayen Islands': 'Longyearbyen', 'Swaziland': 'Mbabane', 'Sweden': 'Stockholm', 'Switzerland': 'Berne', 'Syria': 'Damascus', 'Taiwan': 'Taipei', 'Tajikistan': 'Dushanbe', 'Tanzania': 'Dodoma', 'Thailand': 'Bangkok', 'Togo': 'Lome', 'Tokelau': '', 'Tonga': "Nuku'alofa", 'Trinidad And Tobago': 'Port of Spain', 'Tunisia': 'Tunis', 'Turkey': 'Ankara', 'Turkmenistan': 'Ashgabat', 'Turks And Caicos Islands': 'Cockburn Town', 'Tuvalu': 'Funafuti', 'Uganda': 'Kampala', 'Ukraine': 'Kiev', 'United Arab Emirates': 'Abu Dhabi', 'United Kingdom': 'London', 'United States': 'Washington', 'United States Minor Outlying Islands': '', 'Uruguay': 'Montevideo', 'Uzbekistan': 'Tashkent', 'Vanuatu': 'Port Vila', 'Vatican City State (Holy See)': 'Vatican City', 'Venezuela': 'Caracas', 'Vietnam': 'Hanoi', 'British Virgin Islands': 'Road Town', 'U.S. Virgin Islands': 'Charlotte Amalie', 'USA Virgin Islands': 'Charlotte Amalie', 'Wallis And Futuna Islands': 'Mata Utu', 'Western Sahara': 'El-Aaiun', 'Yemen': 'Sanaa', 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare'}

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

def _expand_address_smart(full_text: str, address_fragment: str, window: int = 80, capitals_names: set|None = None, countries_names: set|None=None) -> str:
    """
    Expand an address fragment by searching nearby for:
        - Postal codes
        - Street numbers
        - Capitals
        - Country names

    Keeps the closest structural signals.
    """
    COUNTRIES_NAMES = set(COUNTRIES_AND_CAPITALS .keys())
    CAPITAL_NAMES = {v for v in COUNTRIES_AND_CAPITALS .values() if v}
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

    capitals = capitals_names if capitals_names is not None else CAPITAL_NAMES
    countries = countries_names if countries_names is not None else COUNTRIES_NAMES

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
            entities_by_label["skills"].append(raw_text)
            continue

        if any(re.search(rf"\b{re.escape(sw)}\b", text_norm) for sw in SOFTWARE_AS_SKILL):
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


