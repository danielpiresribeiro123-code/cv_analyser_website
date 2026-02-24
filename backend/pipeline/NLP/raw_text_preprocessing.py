# pipeline/NLP/raw_text_preprocessing.py

import re
from typing import List


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