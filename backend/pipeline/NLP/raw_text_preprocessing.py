# pipeline/NLP/raw_text_preprocessing.py

import re
from typing import List
from collections import Counter



LETTER_PATTERN = r"[A-Za-zÀ-ÖØ-öø-ÿ]"

COMMON_CV_WORDS = {
    "THE",
    "AND",
    "WITH",
    "MASTER",
    "BACHELOR",
    "DEGREE",
    "SCIENCE",
    "INDUSTRIAL",
    "ENGINEERING",
    "COMPUTER",
    "DATA",
    "GRADUATED",
    "EDUCATION",
    "EXPERIENCE",
    "SKILLS",
    "CORPORATE",
    "PROGRAMMING",
    "MODELING",
    "OTHER",
    "ACTIVITIES",
}

def _space_weight(sep: str) -> int:
    """
    Dá mais peso a separadores visualmente maiores.
    Espaço normal conta 1.
    NBSP (\u00A0) conta 2 porque muitas vezes vem do OCR/PDF como separador mais largo.
    """
    if not sep:
        return 0

    weight = 0
    for ch in sep:
        if ch == "\u00A0":
            weight += 2
        elif ch in {" ", "\t"}:
            weight += 1
    return weight


def _looks_like_spaced_letter_line(line: str) -> bool:
    """
    Deteta linhas OCR com letras isoladas, incluindo casos com hífenes.

    Exemplos:
        G R A D U A T E D
        C O R P O R A T E  E X P E R I E N C E
        N O N - C O R P O R A T E E X P E R I E N C E
    """
    if not line or not line.strip():
        return False

    # contar letras isoladas
    isolated_letters = re.findall(rf"(?<!\w){LETTER_PATTERN}(?!\w)", line)
    if len(isolated_letters) < 4:
        return False

    # permitir apenas letras, espaços e hífenes
    cleaned = re.sub(rf"{LETTER_PATTERN}|\s|-", "", line)
    if cleaned.strip():
        return False

    return True


def fix_spaced_letters(text: str) -> str:
    """
    Reconstrói linhas OCR onde as letras aparecem separadas por espaços,
    incluindo casos com hífenes entre blocos.

    Exemplos:
        G R A D U A T E D  I N  T H E
        -> GRADUATED IN THE

        C O R P O R A T E  E X P E R I E N C E
        -> CORPORATE EXPERIENCE

        N O N - C O R P O R A T E E X P E R I E N C E
        -> NON-CORPORATE EXPERIENCE
    """
    if not text:
        return text

    new_lines = []

    for line in text.splitlines():
        if not _looks_like_spaced_letter_line(line):
            new_lines.append(line)
            continue

        # separar letras, hífenes e grupos de espaços
        parts = re.findall(rf"{LETTER_PATTERN}|-+|[^\S\r\n]+", line)

        if not parts:
            new_lines.append(line)
            continue

        rebuilt = []
        current_word = ""

        i = 0
        while i < len(parts):
            part = parts[i]

            # letra isolada -> acumula
            if re.fullmatch(rf"{LETTER_PATTERN}", part):
                current_word += part

            # hífen -> fecha palavra atual e preserva hífen
            elif re.fullmatch(r"-+", part):
                if current_word:
                    rebuilt.append(current_word)
                    current_word = ""
                rebuilt.append("-")

            # espaços
            elif re.fullmatch(r"[^\S\r\n]+", part):
                weight = _space_weight(part)

                # olhar para próximo token
                next_part = parts[i + 1] if i + 1 < len(parts) else None

                # se o próximo é hífen, não fechar ainda
                if next_part and re.fullmatch(r"-+", next_part):
                    pass
                else:
                    # 2 ou mais espaços -> quebra de palavra
                    if weight >= 2:
                        if current_word:
                            rebuilt.append(current_word)
                            current_word = ""

            i += 1

        if current_word:
            rebuilt.append(current_word)

        # reconstrução final com regras simples
        fixed = []
        for idx, token in enumerate(rebuilt):
            if token == "-":
                # cola hífen à palavra anterior
                if fixed and not fixed[-1].endswith("-"):
                    fixed[-1] = fixed[-1] + "-"
                continue

            # se token anterior acabou em hífen, colar
            if fixed and fixed[-1].endswith("-"):
                fixed[-1] = fixed[-1] + token
            else:
                fixed.append(token)

        fixed_line = " ".join(fixed)
        fixed_line = re.sub(r"\s+", " ", fixed_line).strip()

        new_lines.append(fixed_line)

    return "\n".join(new_lines)

def split_long_ocr_words(text: str) -> str:
    """
    Split long OCR-merged words using a dictionary of common CV words,
    but only on lines that look like full uppercase OCR/header lines.
    """

    def is_uppercase_ocr_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        # tem de ter letras
        if not re.search(rf"{LETTER_PATTERN}", stripped):
            return False

        # se houver minúsculas na linha, não mexer
        if re.search(r"[a-zà-öø-ÿ]", stripped):
            return False

        return True

    def split_word(word: str) -> str:
        if len(word) < 12:
            return word

        if not word.isalpha():
            return word

        upper = word.upper()

        matched_parts = []
        i = 0

        while i < len(upper):
            match = None

            for w in sorted(COMMON_CV_WORDS, key=len, reverse=True):
                if upper.startswith(w, i):
                    match = w
                    break

            if match:
                matched_parts.append(match)
                i += len(match)
            else:
                matched_parts.append(upper[i])
                i += 1

        result = "".join(matched_parts)

        for w in sorted(COMMON_CV_WORDS, key=len, reverse=True):
            result = result.replace(w, f" {w} ")

        result = re.sub(r"\s+", " ", result).strip()

        return result

    new_lines = []

    for line in text.splitlines():
        if not is_uppercase_ocr_line(line):
            new_lines.append(line)
            continue

        tokens = line.split()
        fixed_tokens = [split_word(t) for t in tokens]
        new_lines.append(" ".join(fixed_tokens))

    return "\n".join(new_lines)

def fix_common_merged_headers(text: str) -> str:
    
    replacements = {
        "CORPORATEEXPERIENCE": "CORPORATE EXPERIENCE",
        "NON-CORPORATEEXPERIENCE": "NON-CORPORATE EXPERIENCE",
        "NON - CORPORATEEXPERIENCE": "NON-CORPORATE EXPERIENCE",
        "PROGRAMMINGSKILLSRESUME": "PROGRAMMING SKILLS RESUME",
        "MODELINGSKILLS": "MODELING SKILLS",
        "OTHERACTIVITIES": "OTHER ACTIVITIES",
    }

    for bad, good in replacements.items():
        text = re.sub(rf"\b{bad}\b", good, text, flags=re.IGNORECASE)

    return text

def remove_exact_duplicate_lines(text: str) -> str:
    """
    Remove linhas completamente iguais já encontradas anteriormente.

    Mantém:
        - A primeira ocorrência
        - Ordem original
        - Quebras estruturais

    Ignora:
        - Diferenças apenas de espaços extra
    """

    if not text:
        return text

    seen = set()
    cleaned_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Mantém linhas vazias (preserva estrutura)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Normalização leve (remove múltiplos espaços internos)
        normalized = re.sub(r"\s+", " ", stripped)

        if normalized not in seen:
            seen.add(normalized)
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def add_space_after_colon(text: str) -> str:
    """
    Add a space after ':' when content is glued immediately after it.

    Examples:
        'way:left' -> 'way: left'
        'Email:test@gmail.com' -> 'Email: test@gmail.com'

    Does not affect cases where a space already exists.
    """
    if not text:
        return text

    return re.sub(r":(?=\S)", ": ", text)


def remove_consecutive_duplicate_lines(text: str) -> str:
    """
    Remove duplicated lines appearing consecutively.
    """

    if not text:
        return text

    result = []
    prev = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == prev:
            continue

        result.append(line)
        prev = stripped

    return "\n".join(result)

def remove_near_duplicate_lines(text: str, window: int = 5) -> str:
    """
    Remove linhas quase duplicadas dentro de uma janela de linhas.

    Exemplo removido:
        society.
        society
    """

    if not text:
        return text

    def normalize(line: str) -> str:
        line = line.lower()

        # remover bullets e pontuação leve
        line = re.sub(r"[•\.\,\:\;\-\–\—]", "", line)

        # normalizar espaços
        line = re.sub(r"\s+", " ", line)

        return line.strip()

    lines = text.splitlines()
    result = []

    for i, line in enumerate(lines):

        norm = normalize(line)

        if not norm:
            result.append(line)
            continue

        duplicate = False

        for prev in result[-window:]:
            if normalize(prev) == norm:
                duplicate = True
                break

        if not duplicate:
            result.append(line)

    return "\n".join(result)

def repair_broken_emails(text: str) -> str:
    """
    Join emails broken by newline after '@'.
    """

    return re.sub(
        r'([A-Za-z0-9._%+-]+)\s*@\s*\n\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})',
        r'\1@\2',
        text,
        flags=re.IGNORECASE
    )

def remove_ocr_noise(text: str, min_consecutive: int = 4) -> str:
    """
    Remove blocos de linhas muito curtas quando aparecem consecutivamente,
    o que normalmente indica lixo OCR.

    Mantém linhas curtas isoladas como:
        C
        C+
        C#
        AI
        L1
        L2
    """

    if not text:
        return text

    lines = text.splitlines()
    cleaned = []
    short_block = []

    def is_short_noise(line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        letters = re.findall(r"[A-Za-z]", stripped)

        # muito curta e com pouco conteúdo alfabético
        return len(letters) <= 2 and len(stripped) <= 4

    def flush_block():
        nonlocal short_block, cleaned

        if len(short_block) >= min_consecutive:
            # descarta o bloco inteiro
            short_block = []
        else:
            # mantém porque pode conter coisas úteis
            cleaned.extend(short_block)
            short_block = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_block()
            cleaned.append(line)
            continue

        if is_short_noise(line):
            short_block.append(line)
        else:
            flush_block()
            cleaned.append(line)

    flush_block()

    return "\n".join(cleaned)


SYMBOL_REPLACEMENTS = {
    "📞": " phone ",
    "☎": " phone ",
    "✉": " email ",
    "✉️": " email ",
    "📧": " email ",
    "🔗": " website ",
    "🌐": " website ",
    "📍": " address ",
    "🏠": " address ",
    "💼": " experience ",
}

def replace_symbols_with_labels(text: str) -> str:
    if not text:
        return ""

    for sym, repl in SYMBOL_REPLACEMENTS.items():
        text = text.replace(sym, repl)

    return text


def preprocess_raw_text(text: str):
    if not text:
        return text

    # normalizar NBSP
    text = text.replace("\u00A0", " ")

    text = replace_symbols_with_labels(text)

    text = repair_broken_emails(text)

    text = add_space_after_colon(text)

    text = fix_spaced_letters(text)

    text = split_long_ocr_words(text)

    text = fix_common_merged_headers(text)

    text = remove_ocr_noise(text)

    text = remove_near_duplicate_lines(text)

    text = remove_consecutive_duplicate_lines(text)

    text = remove_exact_duplicate_lines(text)

    return text