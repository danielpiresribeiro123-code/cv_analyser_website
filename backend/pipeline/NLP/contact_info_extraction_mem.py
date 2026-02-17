import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, TypedDict

from utils.path_loader import load_paths

# ==========================================================
# Paths
# ==========================================================

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

TEXT_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"


# ==========================================================
# Regex patterns
# ==========================================================

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_REGEX = re.compile(
    r"""
    (?:
        \+?\d{1,3}[\s\-]?
    )?
    (?:\(?\d{2,3}\)?[\s\-]?)?
    \d{3,4}[\s\-]?\d{3,4}
    """,
    re.VERBOSE,)

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

ADDRESS_REGEX = re.compile(
    r"""
    \b
    \d{1,5}                                  # street number
    \s+
    [A-Za-z0-9\s.,'-]+                        # street name
    (?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)?
    [^\n,]*                                   # rest of line
    (?:,\s*[A-Za-z\s]+){1,3}                  # city/state
    (?:\s*\d{4,6})?                           # postal code
    (?:,\s*[A-Za-z\s]+)?                      # country
    """,
    re.VERBOSE | re.IGNORECASE,
)


EXPERIENCE_REGEX = re.compile(
    r"""
    (?:
        experience|exp\.?|yrs?|years?
    )
    [^\d]{0,20}                # até ~3 palavras no meio
    (\d{1,2})\+?               # número de anos (capturado)
    |
    (\d{1,2})\+?               # número antes
    [^\n]{0,20}
    (?:
        years?|yrs?|experience|exp\.?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ==========================================================
# Text loading (Memory-based)
# ==========================================================

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


# ==========================================================
# Extraction helpers
# ==========================================================

def extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
    """Retorna a primeira ocorrência de um padrão."""
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def extract_all(pattern: re.Pattern, text: str) -> List[str]:
    """Retorna todas as ocorrências únicas de um padrão."""
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
    """Extrai a primeira morada encontrada."""
    match = ADDRESS_REGEX.search(text)
    return match.group(0).strip() if match else None


def extract_experience_years(text: str) -> Optional[int]:
    """
    Extrai anos de experiência a partir de múltiplos formatos de CV.
    Retorna inteiro ou None.
    """
    matches = EXPERIENCE_REGEX.findall(text)

    for m in matches:
        # regex tem dois grupos possíveis
        for g in m:
            if g.isdigit():
                return int(g)

    return None


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
    Extrai informação de contacto e metadados de um texto OCR de CV.
    """

    emails = extract_all(EMAIL_REGEX, text)
    phones = extract_all(PHONE_REGEX, text)
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


def process_folder(
    folder: Path = TEXT_TRAIN_DATA,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Legacy function: processes files from disk.
    For memory-based usage, use process_texts() instead.
    """
    
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    files = list(folder.glob("*.json")) + list(folder.glob("*.txt"))

    if limit:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            # Read from file
            if file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = data.get("text", "")
            elif file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
            
            info = extract_contact_info(text)

            rows.append(
                {
                    "file": file_path.name,
                    **info,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "file": file_path.name,
                    "error": str(e),
                }
            )

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
