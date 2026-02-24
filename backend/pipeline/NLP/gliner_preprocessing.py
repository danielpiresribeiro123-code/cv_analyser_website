import json
import re


BULLET_PATTERN = re.compile(r'^[\u2022\u2023\u25E6\u2043\u2219•\-–]\s+')

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
    # 4. Normalize spaces
    # ----------------------------
    text = re.sub(r"[ \t]+", " ", text)

    return text