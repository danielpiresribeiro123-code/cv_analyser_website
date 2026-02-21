import json
import re

def preprocess_ocr_for_gliner(ocr_json_output):
    """
    Transforms raw OCR output into a semantic text stream optimized for GLiNER.
    
    Methodology:
    1. Splits text into logical blocks (bounding boxes) using double newlines.
    2. Classifies each block as 'Prose' (sentences) or 'List' (skills/items).
    3. Reconstructs text:
       - Prose: Joins lines with SPACE (fixes split entities like "Project\nManager").
       - Lists: Joins lines with COMMA (creates "Python, Java, C++" context).
    """

    # 1. Parse Input
    # Case A: Input is a JSON string -> Parse it
    if isinstance(ocr_json_output, str):
        try:
            data = json.loads(ocr_json_output)
        except json.JSONDecodeError:
            # Fallback: Treat as raw text if not valid JSON
            data = ocr_json_output
    else:
        # Case B: Input is already a Python list/dict
        data = ocr_json_output

    # Normalize to a list of text blocks
    # If data is a single string, split by double newlines to get boxes
    if isinstance(data, str):
        blocks = data.split('\n\n')
    elif isinstance(data, list):
        # Assuming list of strings. If list of dicts, adjust to: block['text']
        blocks = [b if isinstance(b, str) else b.get('text', '') for b in data]
    else:
        raise ValueError("Unsupported input format. Expected JSON string or list.")

    cleaned_blocks = []

    # 2. Iterate and Process Each Block
    for block in blocks:
        if not block.strip():
            continue

        # Remove linhas extremamente curtas (ruído OCR)
        raw_lines = block.split('\n')
        lines = []
        for line in raw_lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # Remove linhas muito pequenas (ex: caracteres soltos)
            if len(clean_line) <= 1:
                continue

            lines.append(clean_line)

        if not lines:
            continue

        first_line_lower = lines[0].lower()

        # ===============================
        # 1 Detect skill blocks only by header
        # ===============================
        is_skill_header = bool(
            re.match(r'^(skills|technologies|stack|competencies|tools)', first_line_lower)
        )

        # ===============================
        # 2 Detect address-like blocks
        # ===============================
        is_address_block = bool(
            re.search(r'\b\d{4}-\d{3}\b', block) or  # Portuguese postal code
            re.search(r'\b(Rua|Avenida|Av\.|Travessa|Praça)\b', block, re.IGNORECASE)
        )

        # ===============================
        # 3 Reconstruction logic
        # ===============================

        if is_skill_header:
            # LIST MODE (only real skills section)

            if not lines[0].endswith(':'):
                lines[0] += ":"

            if len(lines) > 1:
                reconstructed = lines[0] + " " + ", ".join(lines[1:])
            else:
                reconstructed = lines[0]

        elif is_address_block:
            # Always treat address as prose
            reconstructed = " ".join(lines)

        else:
            # Default → prose
            reconstructed = " ".join(lines)

        cleaned_blocks.append(reconstructed)

    # 3. Final Reassembly
    # Join distinct blocks with Double Newlines to preserve section boundaries
    full_text = "\n\n".join(cleaned_blocks)
    
    return full_text

