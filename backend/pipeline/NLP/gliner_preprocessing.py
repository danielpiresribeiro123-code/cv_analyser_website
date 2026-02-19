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

        # Split block into constituent lines
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            continue

        # --- HEURISTIC CLASSIFICATION ---
        
        # Metric 1: Average Line Length (in words)
        # Short lines (< 5 words) usually indicate a List (Skills, Dates).
        # Long lines (> 5 words) usually indicate Prose (Summary, Experience).
        total_words = sum(len(line.split()) for line in lines)
        avg_words = total_words / len(lines)

        # Metric 2: Explicit Header Detection (Strict Targeting)
        # If a block starts with "Skills", it is almost certainly a list.
        first_line_lower = lines[0].lower()
        is_skill_header = bool(re.match(r'^(skills|technologies|stack|competencies|tools)', first_line_lower))

        # --- RECONSTRUCTION LOGIC ---

        if is_skill_header or avg_words < 6:
            # === LIST MODE ===
            # Action: Join with COMMAS to create a dense semantic list.
            # Example: "Skills\nJava\nPython" -> "Skills: Java, Python"
            
            # If it's a header, ensure a colon exists for better NLP context
            if is_skill_header and not lines[0].endswith(':'):
                lines[0] += ":"
            
            # Join remaining items with commas
            # We treat the first line (Header) separately if it was a skill header
            if is_skill_header and len(lines) > 1:
                reconstructed = lines[0] + " " + ", ".join(lines[1:])
            else:
                reconstructed = ", ".join(lines)

        else:
            # === PROSE MODE ===
            # Action: Join with SPACES to fix broken sentences.
            # Example: "Senior Project\nManager" -> "Senior Project Manager"
            reconstructed = " ".join(lines)

        cleaned_blocks.append(reconstructed)

    # 3. Final Reassembly
    # Join distinct blocks with Double Newlines to preserve section boundaries
    full_text = "\n\n".join(cleaned_blocks)
    
    return full_text

