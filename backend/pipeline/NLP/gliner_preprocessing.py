import json
import re


BULLET_PATTERN = re.compile(r'^[\u2022\u2023\u25E6\u2043\u2219•\-–]\s+')
COMMON_UPPER_WORDS = {
    "IN", "ON", "OF", "TO", "BY", "FOR",
    "WITH", "FROM", "THAN", "THAT", "THEN",
    "OFF", "ODD", "ETC", "AND", "OR"
}

def normalize_uppercase_words(text: str) -> str:
    """
    Normalize uppercase words from OCR.

    Rules:
        - words >=5 uppercase letters -> Title case
        - common short uppercase words -> lowercase
    """

    def transform(word: str):

        if word in COMMON_UPPER_WORDS:
            return word.lower()

        if word.isupper() and len(word) >= 5:
            return word.capitalize()

        return word

    tokens = re.split(r'(\W+)', text)

    tokens = [transform(t) if t.isalpha() else t for t in tokens]

    return "".join(tokens)

def separate_uppercase_headers(text: str) -> str:
    """
    Insert blank line between consecutive uppercase OCR headers.
    """

    lines = text.splitlines()
    result = []

    def is_header(line):
        line = line.strip()
        if not line:
            return False

        letters_only = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ ]", "", line).strip()

        return (
            letters_only
            and letters_only.upper() == letters_only
            and len(letters_only.replace(" ", "")) >= 5
        )

    for i, line in enumerate(lines):

        result.append(line)

        if i < len(lines) - 1:

            current = line.strip()
            nxt = lines[i + 1].strip()

            if is_header(current) and is_header(nxt):
                result.append("")

    return "\n".join(result)

PIPE_LIKE_SEPARATORS = r"[|¦·•]"

def normalize_inline_separators(text: str) -> str:
    """
    Normalize OCR separators that should behave like field breaks.
    """

    if not text:
        return text

    # remover separadores no início da linha
    text = re.sub(rf"^\s*{PIPE_LIKE_SEPARATORS}\s*", "", text, flags=re.MULTILINE)

    # remover separadores no fim da linha
    text = re.sub(rf"\s*{PIPE_LIKE_SEPARATORS}\s*$", "", text, flags=re.MULTILINE)

    # separadores internos -> newline
    text = re.sub(rf"\s*{PIPE_LIKE_SEPARATORS}\s*", "\n", text)

    # tratar "/" só quando parece separador entre blocos com espaços
    text = re.sub(r"\s+/\s+", "\n", text)

    # limpar excesso de linhas vazias
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def preprocess_ocr_for_gliner(ocr_json_output):
    """
    Minimal and stable preprocessing for GLiNER.
    Strategy:
        - Remove bullet symbols
        - Fix artificial line breaks
        - Preserve original semantic structure
    """

    # ----------------------------
    # 1. Parse input
    # ----------------------------
    if isinstance(ocr_json_output, str):
        try:
            data = json.loads(ocr_json_output)
        except json.JSONDecodeError:
            data = ocr_json_output
    else:
        data = ocr_json_output

    if isinstance(data, str):
        text = data
    elif isinstance(data, list):
        text = "\n".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in data
        )
    else:
        raise ValueError("Unsupported input format.")

    # ----------------------------
    # 2. Remove bullets
    # ----------------------------
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        line = BULLET_PATTERN.sub("", line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # ----------------------------
    # 3. Fix broken sentence lines
    # ----------------------------
    # If a line does not end with punctuation
    # and next line starts lowercase → join them
    lines = text.split("\n")
    fixed_lines = []
    buffer = ""

    for line in lines:
        if not buffer:
            buffer = line
            continue

        if (
            not buffer.endswith((".", ":", ";"))
            and line
            and line[0].islower()
        ):
            buffer += " " + line
        else:
            fixed_lines.append(buffer)
            buffer = line

    if buffer:
        fixed_lines.append(buffer)

    text = "\n".join(fixed_lines)

    # ----------------------------
    # 4. Normalize inline separators
    # ----------------------------
    text = normalize_inline_separators(text)

    # ----------------------------
    # 5. Normalize uppercase words
    # ----------------------------
    text = normalize_uppercase_words(text)

    # ----------------------------
    # 6. Separate uppercase headers
    # ----------------------------
    text = separate_uppercase_headers(text)

    # ----------------------------
    # 7. Normalize spaces
    # ----------------------------
    text = re.sub(r"[ \t]+", " ", text)

    return text



def remove_known_contacts_from_text(
    text: str,
    emails: list[str] | None = None,
    phones: list[str] | None = None,
    websites: list[str] | None = None,
) -> str:
    """
    Remove from text the known emails, phones and websites already extracted by regex.

    For phones, supports OCR/display variants with optional spaces,
    parentheses and separators between digits.

    Examples matched from normalized phone '+351933068068':
        +351933068068
        +351 933068068
        +351 933 068 068
        (+351) 933068068
        (+351) 933 068 068
    """
    if not text:
        return text

    cleaned = text

    def _remove_literal_items(source_text: str, items: list[str] | None) -> str:
        if not items:
            return source_text

        for item in sorted(
            {i.strip() for i in items if i and i.strip()},
            key=len,
            reverse=True,
        ):
            source_text = re.sub(re.escape(item), " ", source_text, flags=re.IGNORECASE)

        return source_text

    def _build_phone_pattern(phone: str) -> str | None:
        """
        Build a flexible regex from a normalized phone number.

        Keeps digits mandatory, but allows optional spaces, hyphens,
        dots and parentheses around/inside the visible representation.
        """
        if not phone:
            return None

        phone = phone.strip()
        if not phone:
            return None

        has_plus = phone.startswith("+")
        digits = re.sub(r"\D", "", phone)

        if len(digits) < 6:
            return None

        sep = r"[\s()./-]*"

        if has_plus:
            pattern = r"(?:\+|\(\+)\s*" + sep.join(map(re.escape, digits)) + r"\)?"
        else:
            pattern = sep.join(map(re.escape, digits))

        return pattern

    def _remove_phone_items(source_text: str, items: list[str] | None) -> str:
        if not items:
            return source_text

        unique_items = sorted(
            {i.strip() for i in items if i and i.strip()},
            key=len,
            reverse=True,
        )

        for item in unique_items:
            pattern = _build_phone_pattern(item)
            if not pattern:
                continue

            source_text = re.sub(pattern, " ", source_text, flags=re.IGNORECASE)

        return source_text

    def _build_website_pattern(url: str) -> str | None:
        """
        Build a regex pattern that matches multiple visible variants of the same website.

        Example stored:
            https://www.indiehackers.com/Dordio

        Should match:
            https://www.indiehackers.com/Dordio
            http://www.indiehackers.com/Dordio
            www.indiehackers.com/Dordio
            indiehackers.com/Dordio
            https://www.indiehackers.com
            www.indiehackers.com
            indiehackers.com
        """
        if not url:
            return None

        url = url.strip()
        if not url:
            return None

        url_no_proto = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)

        parts = url_no_proto.split("/", 1)
        domain = parts[0]
        path = "/" + parts[1] if len(parts) > 1 else ""

        domain = re.sub(r"^www\.", "", domain, flags=re.IGNORECASE)

        domain = re.escape(domain)
        path = re.escape(path)

        pattern = rf"(?:https?://)?(?:www\.)?{domain}(?:{path})?"

        return pattern

    def _remove_website_items(source_text: str, items: list[str] | None) -> str:
        if not items:
            return source_text

        unique_items = sorted(
            {i.strip() for i in items if i and i.strip()},
            key=len,
            reverse=True,
        )

        for item in unique_items:
            pattern = _build_website_pattern(item)
            if not pattern:
                continue

            source_text = re.sub(pattern, " ", source_text, flags=re.IGNORECASE)

        return source_text

    cleaned = _remove_literal_items(cleaned, emails)
    cleaned = _remove_phone_items(cleaned, phones)
    cleaned = _remove_website_items(cleaned, websites)

    # limpeza leve
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()