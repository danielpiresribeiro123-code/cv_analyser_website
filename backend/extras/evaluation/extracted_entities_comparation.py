import json
import re
from pathlib import Path
from collections import Counter
from typing import Iterable, Any


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
from services.paths_models_and_data import get_gliner_model, LANGUAGES
from pipeline.NLP.entity_extraction_gliner import process_texts_gliner
from pipeline.NLP.contact_extraction_regex import extract_contact_info, extract_languages
from pipeline.NLP.gliner_preprocessing import (
    preprocess_ocr_for_gliner,
    remove_known_contacts_from_text,
)
from services.paths_models_and_data import (EXTRAS_DIR)

EVALUATION_TXT_DIR = EXTRAS_DIR / "evaluation" / "txt"

languages_set = {
    re.sub(r"\s+", " ", str(lang).strip()).casefold()
    for lang in LANGUAGES
    if isinstance(lang, str) and str(lang).strip()
}



# ----------------------------------------------------------
# Main function
# ----------------------------------------------------------

def compare_dataset_entities_vs_gliner(
    json_folder: str,
    start: int,
    end: int,
    dataset_output_file: str | Path = "dataset_entities.txt",
    gliner_output_file: str | Path = "gliner_entities.txt",
    regex_output_file: str | Path = "regex_languages.txt",
    similar_output_file: str | Path = "similar_entities.txt",
    different_output_file: str | Path = "different_entities.txt",
    dataset_only_output_file: str | Path = "only_dataset_entities.txt",
    gliner_only_output_file: str | Path = "only_gliner_entities.txt",
    summary_output_file: str | Path | None = "summary_report.txt",
    exact_stopwords: Iterable[str] | None = None,
    partial_stopwords: Iterable[str] | None = None,
    trusted_text_entities_as_true_positive: Iterable[str] | None = None,
    ignore_case: bool = True,
    normalize_spaces: bool = True,
    min_length: int = 2,
    sort_outputs: bool = True,
    sort_by: str = "frequency",
    gliner_threshold: float = 0.15,
    gliner_skills_column: str = "skills",
    gliner_job_titles_column: str = "job_title",
    enable_regex_language_extraction: bool = True,
    trust_regex_languages_as_true_positive: bool = True,
    min_frequency_filters: dict[str, int] | None = None,
    gliner_model: Any | None = None,
):
    """
    Compare dataset entities (stored under the "SKILL:" annotation prefix)
    against entities extracted by the GLiNER pipeline on a per-document basis.

    Dataset side:
    - All valid annotations under "SKILL:"

    GLiNER side:
    - skills
    - job_title
    - regex-extracted languages (optional)

    Matching is performed per document, not globally across all CVs.

    Generated files:
    - dataset_output_file: unique dataset entities + frequency
    - gliner_output_file: unique GLiNER entities (+ regex languages) + frequency
    - regex_output_file: unique regex-language entities only + frequency
    - similar_output_file: matched dataset <-> GLiNER pairs + frequencies
    - different_output_file: grouped unmatched entities from both sides + frequencies
    - dataset_only_output_file: unmatched dataset entities + frequency
    - gliner_only_output_file: unmatched GLiNER entities (+ regex languages) + frequency

    Printed metrics:
    - unique and total counts
    - intersection percentage
    - precision
    - recall
    - F1 score
    """

    if gliner_model is None:
        gliner_model = get_gliner_model()

    folder = Path(json_folder)

    if exact_stopwords is None:
        exact_stopwords = set()
    else:
        exact_stopwords = set(exact_stopwords)

    if partial_stopwords is None:
        partial_stopwords = set()
    else:
        partial_stopwords = set(partial_stopwords)

    if trusted_text_entities_as_true_positive is None:
        trusted_text_entities_as_true_positive = []
    else:
        trusted_text_entities_as_true_positive = [
            str(x).strip()
            for x in trusted_text_entities_as_true_positive
            if isinstance(x, str) and str(x).strip()
        ]

    def normalize_entity(entity: str) -> str:
        entity = entity.strip()

        if normalize_spaces:
            entity = re.sub(r"\s+", " ", entity)

        if ignore_case:
            entity = entity.casefold()

        return entity

    def tokenize_entity(entity: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", normalize_entity(entity)))
    
    def is_language_entity(entity: str) -> bool:
        if not entity:
            return False

        entity_norm = normalize_entity(entity)
        return entity_norm in languages_set

    def is_valid_entity(entity_original: str) -> bool:
        if not entity_original:
            return False

        if len(entity_original.strip()) < min_length:
            return False

        entity_norm = normalize_entity(entity_original)

        if entity_norm in normalized_exact_stopwords:
            return False

        if any(stop_partial in entity_norm for stop_partial in normalized_partial_stopwords):
            return False

        return True

    def extract_dataset_entities(annotations: list) -> list[str]:
        entities = []

        for annotation in annotations:
            if len(annotation) < 3:
                continue

            _, _, label = annotation

            if not isinstance(label, str):
                continue

            if not label.startswith("SKILL:"):
                continue

            entity = label.replace("SKILL:", "", 1).strip()

            if is_valid_entity(entity):
                entities.append(entity)

        return entities

    def extract_entities_from_row(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str) and is_valid_entity(item):
                    result.append(item.strip())
            return result

        if isinstance(value, str):
            parts = re.split(r"[,\n;]+", value)
            return [part.strip() for part in parts if is_valid_entity(part.strip())]

        return []
    
    def entity_appears_in_text(entity: str, text: str) -> bool:
        if not entity or not text:
            return False

        entity_norm = normalize_entity(entity)
        text_norm = normalize_entity(text)

        # Se tiver símbolos como +, #, ., -, /, não uses \b
        if re.search(r"[^\w\s]", entity_norm):
            return entity_norm in text_norm

        pattern = rf"\b{re.escape(entity_norm)}\b"
        return bool(re.search(pattern, text_norm, re.IGNORECASE))

    def deduplicate_preserving_order(values: list[str]) -> list[str]:
        seen = set()
        result = []

        for value in values:
            norm = normalize_entity(value)
            if norm not in seen:
                seen.add(norm)
                result.append(value.strip())

        return result

    def sort_values(
            values: list[str],
            counts: Counter | None = None,
        ) -> list[str]:
            if not sort_outputs:
                return values

            mode = sort_by.strip().casefold()

            if mode == "frequency" and counts is not None:
                return sorted(
                    values,
                    key=lambda x: (-counts.get(normalize_entity(x), 0), x.casefold())
                )

            return sorted(values, key=lambda x: x.casefold())

    def split_entity_tokens(entity: str) -> list[str]:
        if not entity:
            return []

        entity = normalize_entity(entity)
        tokens = re.split(r"[^a-zA-Z0-9#+.]+", entity)
        return [t for t in tokens if t]


    def generate_token_variants(token: str) -> set[str]:
        """
        Gera variantes simples/flexíveis para matching:
        - plural/singular
        - terminações comuns
        - pequenas normalizações OCR
        """
        if not token:
            return set()

        t = normalize_entity(token)
        variants = {t}

        # plural/singular simples
        if len(t) > 2:
            variants.add(t + "s")
            variants.add(t + "es")

        if len(t) > 3 and t.endswith("s"):
            variants.add(t[:-1])

        if len(t) > 4 and t.endswith("es"):
            variants.add(t[:-2])

        if len(t) > 4 and t.endswith("ies"):
            variants.add(t[:-3] + "y")

        if len(t) > 3 and t.endswith("y"):
            variants.add(t[:-1] + "ies")

        # terminações comuns
        suffix_rules = [
            ("ing", ""),
            ("ing", "e"),
            ("er", ""),
            ("or", ""),
            ("ion", ""),
            ("ion", "e"),
            ("ive", ""),
            ("ive", "ion"),
            ("al", ""),
            ("ed", ""),
            ("ment", ""),
            ("ness", ""),
            ("ity", ""),
            ("ical", "ic"),
            ("ic", ""),
            ("ian", ""),
            ("ist", ""),
            ("ize", ""),
            ("ise", ""),
        ]

        for suffix, replacement in suffix_rules:
            if len(t) > len(suffix) + 2 and t.endswith(suffix):
                variants.add(t[:-len(suffix)] + replacement)

        # casos úteis para engineer / engineering
        if t.endswith("eer"):
            variants.add(t[:-3])
            variants.add(t[:-3] + "ing")

        if t.endswith("ering"):
            variants.add(t[:-5] + "eer")
            variants.add(t[:-3])

        # OCR / separações comuns
        variants.add(t.replace(" ", ""))
        variants.add(t.replace(".", ""))
        variants.add(t.replace("-", ""))
        variants.add(t.replace("/", ""))

        return {v for v in variants if v and len(v) >= 2}

    
    def token_variant_match(token_a: str, token_b: str) -> bool:
        va = generate_token_variants(token_a)
        vb = generate_token_variants(token_b)

        if va & vb:
            return True

        for a in va:
            for b in vb:
                if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
                    return True

                if is_one_edit_away_for_long_strings(a, b, min_len=7):
                    return True

        return False

    def is_one_edit_away_for_long_strings(a: str, b: str, min_len: int = 7) -> bool:
        """
        True se duas strings tiverem distância de edição <= 1
        e pelo menos uma delas tiver comprimento >= min_len.

        Casos aceites:
        - 1 char removido
        - 1 char adicionado
        - 1 char substituído
        """
        a = normalize_entity(a)
        b = normalize_entity(b)

        if not a or not b:
            return False

        if a == b:
            return True

        if max(len(a), len(b)) < min_len:
            return False

        len_a = len(a)
        len_b = len(b)

        # Se diferença de tamanho for > 1, já não pode ser 1 edição
        if abs(len_a - len_b) > 1:
            return False

        # Caso mesmo tamanho -> no máximo 1 substituição
        if len_a == len_b:
            diff = sum(1 for ca, cb in zip(a, b) if ca != cb)
            return diff <= 1

        # Caso diferença de 1 -> 1 inserção/remoção
        # garantir que a é a menor
        if len_a > len_b:
            a, b = b, a
            len_a, len_b = len_b, len_a

        i = 0
        j = 0
        found_difference = False

        while i < len_a and j < len_b:
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                if found_difference:
                    return False
                found_difference = True
                j += 1  # salta um char na maior

        return True

    def entity_variant_substring_match(entity_a: str, entity_b: str) -> tuple[bool, float, str]:
        """
        Faz matching mais flexível:
        - exact
        - 1-char error para strings longas
        - substring direto
        - token variants
        """
        a = normalize_entity(entity_a)
        b = normalize_entity(entity_b)

        if not a or not b:
            return False, 0.0, "none"

        if a == b:
            return True, 1.0, "exact"

        # novo: tolerância a 1 erro para strings maiores
        if is_one_edit_away_for_long_strings(a, b, min_len=7):
            shorter = min(len(a), len(b))
            longer = max(len(a), len(b))
            base_score = shorter / longer if longer else 0.0
            return True, max(0.88, base_score), "one_edit_long"

        if a in b or b in a:
            shorter = min(len(a), len(b))
            longer = max(len(a), len(b))
            score = shorter / longer if longer else 0.0
            return True, max(0.80, score), "substring"

        tokens_a = split_entity_tokens(a)
        tokens_b = split_entity_tokens(b)

        if not tokens_a or not tokens_b:
            return False, 0.0, "none"

        matched_a = sum(
            1 for ta in tokens_a
            if any(token_variant_match(ta, tb) for tb in tokens_b)
        )

        matched_b = sum(
            1 for tb in tokens_b
            if any(token_variant_match(tb, ta) for ta in tokens_a)
        )

        score_a = matched_a / len(tokens_a) if tokens_a else 0.0
        score_b = matched_b / len(tokens_b) if tokens_b else 0.0
        best_score = max(score_a, score_b)

        if best_score >= 1.0:
            return True, 0.93, "variant_token_full"

        if best_score >= 0.67:
            return True, 0.82, "variant_token_partial"

        return False, 0.0, "none"

    def are_similar(entity_a: str, entity_b: str) -> tuple[bool, float, str]:
        a = normalize_entity(entity_a)
        b = normalize_entity(entity_b)

        if a == b:
            return True, 1.0, "exact"

        # 1) matching flexível por substring + variantes
        variant_match, variant_score, variant_type = entity_variant_substring_match(a, b)
        if variant_match:
            return variant_match, variant_score, variant_type

        tokens_a = tokenize_entity(a)
        tokens_b = tokenize_entity(b)

        if not tokens_a or not tokens_b:
            return False, 0.0, "none"

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        smaller = min(len(tokens_a), len(tokens_b))

        # 2) token overlap clássico
        score_small = intersection / smaller if smaller > 0 else 0.0
        score_jaccard = intersection / union if union > 0 else 0.0

        if score_small >= (2 / 3):
            score = max(score_small, score_jaccard)
            return True, score, "token_overlap"

        return False, 0.0, "none"

    def match_type_priority(match_type: str) -> int:
        priority = {
            "exact": 6,
            "one_edit_long": 5,
            "variant_token_full": 4,
            "substring": 3,
            "variant_token_partial": 2,
            "token_overlap": 1,
            "regex_language_override": 1,
            "trusted_text_entity_override": 1,
            "none": 0,
        }
        return priority.get(match_type, 0)

    def save_list_with_frequency(
        file_path: str|Path,
        values: list[str],
        counts: Counter,
        title: str | None = None,
    ) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        values = sort_values(values, counts)

        with open(path, "w", encoding="utf-8") as f:
            if title:
                f.write(f"{title}\n")
                f.write("=" * len(title) + "\n\n")

            for value in values:
                norm = normalize_entity(value)
                frequency = counts.get(norm, 0)
                f.write(f"{value}\t{frequency}\n")

    def save_similar_with_frequency(
        file_path: str|Path,
        similar_items: list[tuple[str, str, float, str]],
        dataset_counts: Counter,
        gliner_counts: Counter,
    ) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = sort_by.strip().casefold()

        if sort_outputs:
            if mode == "frequency":
                similar_items = sorted(
                    similar_items,
                    key=lambda x: (
                        -(dataset_counts.get(normalize_entity(x[0]), 0) +
                          gliner_counts.get(normalize_entity(x[1]), 0)),
                        x[0].casefold(),
                        x[1].casefold(),
                    ),
                )
            else:
                similar_items = sorted(
                    similar_items,
                    key=lambda x: (x[0].casefold(), x[1].casefold()),
                )

        with open(path, "w", encoding="utf-8") as f:
            f.write("SIMILAR_ENTITIES\n")
            f.write("================\n\n")

            for dataset_entity, gliner_entity, score, match_type in similar_items:
                dataset_freq = dataset_counts.get(normalize_entity(dataset_entity), 0)
                gliner_freq = gliner_counts.get(normalize_entity(gliner_entity), 0)
                f.write(
                    f"{dataset_entity}\t{dataset_freq}\t<->\t"
                    f"{gliner_entity}\t{gliner_freq}\t"
                    f"[score={score:.2f}; match_type={match_type}]\n"
                )

    def save_different_groups(
        file_path: str|Path,
        dataset_only: list[str],
        gliner_only: list[str],
        dataset_counts: Counter,
        gliner_counts: Counter,
    ) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dataset_only = sort_values(dataset_only, dataset_counts)
        gliner_only = sort_values(gliner_only, gliner_counts)

        with open(path, "w", encoding="utf-8") as f:
            f.write("ONLY_DATASET\n")
            f.write("============\n")
            for entity in dataset_only:
                frequency = dataset_counts.get(normalize_entity(entity), 0)
                f.write(f"{entity}\t{frequency}\n")

            f.write("\nONLY_GLINER_ENTITIES_PLUS_REGEX_LANGUAGES\n")
            f.write("=========================================\n")
            for entity in gliner_only:
                frequency = gliner_counts.get(normalize_entity(entity), 0)
                f.write(f"{entity}\t{frequency}\n")

    def filter_maps_by_min_frequency(
        entity_map: dict[str, str],
        counts: Counter,
        min_frequency: int,
    ) -> tuple[dict[str, str], Counter]:
        if min_frequency < 0:
            return dict(entity_map), Counter(counts)

        filtered_map = {
            norm: original
            for norm, original in entity_map.items()
            if counts.get(norm, 0) > min_frequency
        }

        filtered_counts = Counter({
            norm: freq
            for norm, freq in counts.items()
            if norm in filtered_map
        })

        return filtered_map, filtered_counts
    
    def save_summary_report(
        file_path: str|Path,
        summary_data: dict,
    ) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("SUMMARY REPORT\n")
            f.write("==============\n\n")

            f.write("EXECUTION INFO\n")
            f.write("--------------\n")
            f.write(f"JSON folder: {summary_data['json_folder']}\n")
            f.write(f"Start index: {summary_data['start']}\n")
            f.write(f"End index: {summary_data['end']}\n")
            f.write(f"Requested CVs: {summary_data['requested_cvs']}\n")
            f.write(f"Existing files found: {summary_data['existing_files']}\n")
            f.write(f"Processed CVs: {summary_data['processed_cvs']}\n")
            f.write(f"Missing files: {summary_data['missing_files_count']}\n")
            f.write(f"GLiNER / regex errors: {summary_data['gliner_errors_count']}\n\n")

            f.write("ENTITY COUNTS\n")
            f.write("-------------\n")
            f.write(f"Unique dataset entities: {summary_data['total_dataset_unique']}\n")
            f.write(f"Total dataset entity occurrences: {summary_data['total_dataset_occurrences']}\n")
            f.write(f"Unique GLiNER entities + regex languages: {summary_data['total_gliner_unique']}\n")
            f.write(f"Total GLiNER entity occurrences + regex languages: {summary_data['total_gliner_occurrences']}\n")
            f.write(f"Unique regex language-only entities: {summary_data['total_regex_unique']}\n")
            f.write(f"Total regex language-only occurrences: {summary_data['total_regex_occurrences']}\n")
            f.write(f"Matched entity occurrences (including overrides): {summary_data['total_intersection']}\n")
            f.write(f"Entities only in dataset: {summary_data['dataset_only_count']}\n")
            f.write(f"Entities only in GLiNER + regex languages: {summary_data['gliner_only_count']}\n\n")

            f.write("METRICS\n")
            f.write("-------\n")
            f.write(f"Intersection percentage: {summary_data['intersection_percentage']:.2f}%\n")
            f.write(f"Precision: {summary_data['precision']:.2f}%\n")
            f.write(f"Recall: {summary_data['recall']:.2f}%\n")
            f.write(f"F1 score: {summary_data['f1']:.2f}%\n\n")

            if summary_data["missing_files"]:
                f.write("MISSING FILES\n")
                f.write("-------------\n")
                for item in summary_data["missing_files"]:
                    f.write(f"{item}\n")
                f.write("\n")

            if summary_data["gliner_errors"]:
                f.write("GLINER / REGEX ERRORS\n")
                f.write("---------------------\n")
                for item in summary_data["gliner_errors"]:
                    f.write(f"{item}\n")
                f.write("\n")

    sort_by_normalized = sort_by.strip().casefold()
    if sort_by_normalized not in {"alphabetical", "frequency"}:
        raise ValueError("sort_by must be 'alphabetical' or 'frequency'")

    normalized_exact_stopwords = {normalize_entity(s) for s in exact_stopwords}
    normalized_partial_stopwords = {normalize_entity(s) for s in partial_stopwords}

    trusted_text_entities_map = {
        normalize_entity(entity): entity
        for entity in trusted_text_entities_as_true_positive
        if is_valid_entity(entity)
    }

    if min_frequency_filters is None:
        min_frequency_filters = {}

    dataset_min_freq = int(min_frequency_filters.get("dataset", -1))
    gliner_min_freq = int(min_frequency_filters.get("gliner", -1))
    regex_min_freq = int(min_frequency_filters.get("regex", -1))

    # Global collections
    all_dataset_map: dict[str, str] = {}
    all_gliner_map: dict[str, str] = {}
    regex_map: dict[str, str] = {}

    dataset_only_map: dict[str, str] = {}
    gliner_only_map: dict[str, str] = {}

    dataset_counts = Counter()
    gliner_counts = Counter()
    regex_counts = Counter()

    dataset_only_counts = Counter()
    gliner_only_counts = Counter()

    similar_info: list[tuple[str, str, float, str]] = []

    missing_files = []
    gliner_errors = []

    total_dataset_matched = 0
    total_gliner_matched = 0

    requested_cvs = max(0, end - start + 1)
    processed_cvs = 0
    existing_files = 0

    gliner_error_files = set()

    for i in range(start, end + 1):
        file_name = f"cv ({i})_annotated.json"
        path = folder / file_name

        if not path.exists():
            missing_files.append(file_name)
            continue

        existing_files += 1

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_cvs += 1

        text = data.get("text", "")
        annotations = data.get("annotations", [])

        # -------------------------
        # Dataset entities (current document)
        # -------------------------
        dataset_entities = extract_dataset_entities(annotations)

        dataset_doc_map: dict[str, str] = {}
        for entity in dataset_entities:
            norm = normalize_entity(entity)

            if norm not in dataset_doc_map:
                dataset_doc_map[norm] = entity.strip()

            if norm not in all_dataset_map:
                all_dataset_map[norm] = entity.strip()

            dataset_counts[norm] += 1

        # -------------------------
        # GLiNER entities (current document)
        # -------------------------
        try:
            contact_info = extract_contact_info(text)

            gliner_input_text = remove_known_contacts_from_text(
                text,
                emails=contact_info.get("all_emails"),
                phones=contact_info.get("all_phones"),
                websites=contact_info.get("all_websites"),
            )

            preprocessed_text = preprocess_ocr_for_gliner(gliner_input_text)

            gliner_df = process_texts_gliner(
                texts=[{
                    "gliner_text": preprocessed_text,
                    "raw_text": text,
                }],
                model=gliner_model,
                gliner_threshold=gliner_threshold,
                debug_save_json=False,
                debug_output_path=None,
            )

            gliner_skills = []
            gliner_job_titles = []

            if not gliner_df.empty:
                if gliner_skills_column in gliner_df.columns:
                    gliner_skills_value = gliner_df.iloc[0].get(gliner_skills_column)
                    gliner_skills = extract_entities_from_row(gliner_skills_value)
                    gliner_skills = [
                        entity for entity in gliner_skills
                        if not is_language_entity(entity)
                    ]

                if gliner_job_titles_column in gliner_df.columns:
                    gliner_job_titles_value = gliner_df.iloc[0].get(gliner_job_titles_column)
                    gliner_job_titles = extract_entities_from_row(gliner_job_titles_value)
                    gliner_job_titles = [
                        entity for entity in gliner_job_titles
                        if not is_language_entity(entity)
                    ]

        except Exception as e:
            gliner_errors.append(f"{file_name}: {e}")
            gliner_error_files.add(file_name)
            gliner_skills = []
            gliner_job_titles = []

        # -------------------------
        # Regex languages (current document)
        # -------------------------
        regex_languages = []
        if enable_regex_language_extraction:
            try:
                regex_languages = extract_languages(text)
                regex_languages = [
                    lang.strip()
                    for lang in regex_languages
                    if is_valid_entity(lang.strip())
                ]
            except Exception as e:
                gliner_errors.append(f"{file_name} [languages_regex]: {e}")
                regex_languages = []

        for entity in regex_languages:
            norm = normalize_entity(entity)
            if norm not in regex_map:
                regex_map[norm] = entity.strip()
            regex_counts[norm] += 1

        # -------------------------
        # Merge GLiNER skills + job titles + regex languages
        # -------------------------
        gliner_non_language_entities = deduplicate_preserving_order(
            gliner_skills + gliner_job_titles
        )

        gliner_total_entities = deduplicate_preserving_order(
            gliner_non_language_entities + regex_languages
        )

        gliner_doc_map: dict[str, str] = {}
        for entity in gliner_total_entities:
            norm = normalize_entity(entity)

            if norm not in gliner_doc_map:
                gliner_doc_map[norm] = entity.strip()

            if norm not in all_gliner_map:
                all_gliner_map[norm] = entity.strip()

            gliner_counts[norm] += 1

        # -------------------------
        # Per-document matching
        # -------------------------
        dataset_doc_matched = set()
        gliner_doc_matched = set()

        doc_candidate_matches: list[tuple[str, str, float, str]] = []

        for norm_dataset, dataset_entity in dataset_doc_map.items():
            for norm_gliner, gliner_entity in gliner_doc_map.items():
                match, score, match_type = are_similar(dataset_entity, gliner_entity)

                if match:
                    doc_candidate_matches.append(
                        (norm_dataset, norm_gliner, score, match_type)
                    )

        doc_candidate_matches = sorted(
            doc_candidate_matches,
            key=lambda x: (
                -match_type_priority(x[3]),
                -x[2],
                abs(len(dataset_doc_map[x[0]]) - len(gliner_doc_map[x[1]])),
                -(dataset_counts.get(x[0], 0) + gliner_counts.get(x[1], 0)),
                dataset_doc_map[x[0]].casefold(),
                gliner_doc_map[x[1]].casefold(),
            )
        )

        for norm_dataset, norm_gliner, score, match_type in doc_candidate_matches:
            if norm_dataset in dataset_doc_matched:
                continue
            if norm_gliner in gliner_doc_matched:
                continue

            similar_info.append(
                (
                    dataset_doc_map[norm_dataset],
                    gliner_doc_map[norm_gliner],
                    score,
                    match_type,
                )
            )
            dataset_doc_matched.add(norm_dataset)
            gliner_doc_matched.add(norm_gliner)

        # -------------------------
        # Optional rule:
        # trust regex languages as true positives even if missing in dataset
        # and count them as virtual dataset matches too
        # -------------------------
        if trust_regex_languages_as_true_positive:
            for entity in regex_languages:
                norm_lang = normalize_entity(entity)

                if norm_lang not in gliner_doc_map:
                    continue

                if norm_lang in gliner_doc_matched:
                    continue

                # se o dataset já tiver essa língua, o matching normal trata disso
                if norm_lang in dataset_doc_map:
                    continue

                gliner_doc_matched.add(norm_lang)

                pseudo_dataset_entity = f"[DATASET_MISSING_LANGUAGE::{gliner_doc_map[norm_lang]}]"

                similar_info.append(
                    (
                        pseudo_dataset_entity,
                        gliner_doc_map[norm_lang],
                        1.0,
                        "regex_language_override",
                    )
                )

        # -------------------------
        # Optional rule:
        # trust selected text entities as true positives even if missing in dataset
        # -------------------------
        if trusted_text_entities_map:
            for norm_trusted, original_trusted in trusted_text_entities_map.items():
                if not entity_appears_in_text(original_trusted, text):
                    continue

                best_norm_gliner = None
                best_score = 0.0
                best_match_type = "none"

                for norm_gliner, gliner_entity in gliner_doc_map.items():
                    if norm_gliner in gliner_doc_matched:
                        continue

                    match, score, match_type = are_similar(original_trusted, gliner_entity)
                    if not match:
                        continue

                    if (
                        score > best_score or
                        (
                            score == best_score and
                            match_type_priority(match_type) > match_type_priority(best_match_type)
                        )
                    ):
                        best_norm_gliner = norm_gliner
                        best_score = score
                        best_match_type = match_type

                if best_norm_gliner is None:
                    continue

                if best_norm_gliner in gliner_doc_matched:
                    continue

                gliner_doc_matched.add(best_norm_gliner)

                pseudo_dataset_entity = (
                    f"[DATASET_MISSING_TRUSTED_ENTITY::{gliner_doc_map[best_norm_gliner]}]"
                )

                similar_info.append(
                    (
                        pseudo_dataset_entity,
                        gliner_doc_map[best_norm_gliner],
                        max(0.95, best_score),
                        "trusted_text_entity_override",
                    )
                )

        total_dataset_matched += len(dataset_doc_matched)
        total_gliner_matched += len(gliner_doc_matched)

        # -------------------------
        # Unmatched entities per document
        # -------------------------
        for norm_dataset, entity in dataset_doc_map.items():
            if norm_dataset not in dataset_doc_matched:
                if norm_dataset not in dataset_only_map:
                    dataset_only_map[norm_dataset] = entity
                dataset_only_counts[norm_dataset] += 1

        for norm_gliner, entity in gliner_doc_map.items():
            if norm_gliner not in gliner_doc_matched:
                if norm_gliner not in gliner_only_map:
                    gliner_only_map[norm_gliner] = entity
                gliner_only_counts[norm_gliner] += 1


    # -------------------------
    # Apply optional min-frequency filtering
    # Removes entities with frequency <= configured threshold
    # -------------------------
    if dataset_min_freq >= 0:
        dataset_only_map = {
            norm: entity
            for norm, entity in dataset_only_map.items()
            if dataset_counts.get(norm, 0) > dataset_min_freq
        }
        dataset_only_counts = Counter({
            norm: freq
            for norm, freq in dataset_only_counts.items()
            if norm in dataset_only_map
        })

    if gliner_min_freq >= 0:
        gliner_only_map = {
            norm: entity
            for norm, entity in gliner_only_map.items()
            if gliner_counts.get(norm, 0) > gliner_min_freq
        }
        gliner_only_counts = Counter({
            norm: freq
            for norm, freq in gliner_only_counts.items()
            if norm in gliner_only_map
        })

    dataset_unique_entities = list(all_dataset_map.values())
    gliner_unique_entities = list(all_gliner_map.values())
    regex_unique_entities = list(regex_map.values())

    dataset_only_entities = list(dataset_only_map.values())
    gliner_only_entities = list(gliner_only_map.values())

    total_dataset_entity_occurrences = sum(dataset_counts.values())
    total_gliner_entity_occurrences = sum(gliner_counts.values())
    total_regex_entity_occurrences = sum(regex_counts.values())
    

    # -------------------------
    # Save output files
    # -------------------------
    save_list_with_frequency(
        dataset_output_file,
        dataset_unique_entities,
        dataset_counts,
        title="DATASET_ENTITIES",
    )

    save_list_with_frequency(
        gliner_output_file,
        gliner_unique_entities,
        gliner_counts,
        title="GLINER_PLUS_REGEX_LANGUAGE_ENTITIES",
    )

    save_list_with_frequency(
        regex_output_file,
        regex_unique_entities,
        regex_counts,
        title="REGEX_LANGUAGE_ENTITIES_ONLY",
    )

    save_similar_with_frequency(
        similar_output_file,
        similar_info,
        dataset_counts,
        gliner_counts,
    )

    save_list_with_frequency(
        dataset_only_output_file,
        dataset_only_entities,
        dataset_only_counts,
        title="ENTITIES_ONLY_IN_DATASET",
    )

    save_list_with_frequency(
        gliner_only_output_file,
        gliner_only_entities,
        gliner_only_counts,
        title="ENTITIES_ONLY_IN_GLINER_PLUS_REGEX_LANGUAGES",
    )

    save_different_groups(
        different_output_file,
        dataset_only_entities,
        gliner_only_entities,
        dataset_only_counts,
        gliner_only_counts,
    )

    # -------------------------
    # Global metrics
    # -------------------------
    total_dataset_unique = len(all_dataset_map)
    total_gliner_unique = len(all_gliner_map)
    total_regex_unique = len(regex_map)

    total_dataset_occurrences = total_dataset_entity_occurrences
    total_gliner_occurrences = total_gliner_entity_occurrences

    total_dataset_matched = len(similar_info)
    total_gliner_matched = len(similar_info)

    total_intersection = len(similar_info)
    precision = (total_gliner_matched / total_gliner_occurrences * 100) if total_gliner_occurrences > 0 else 0.0
    recall = (total_dataset_matched / total_dataset_occurrences * 100) if total_dataset_occurrences > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    intersection_percentage = (
        total_intersection / (
            total_dataset_occurrences + total_gliner_occurrences - total_intersection
        ) * 100
    ) if (total_dataset_occurrences + total_gliner_occurrences - total_intersection) > 0 else 0.0


    summary_data = {
        "json_folder": str(folder),
        "start": start,
        "end": end,
        "requested_cvs": requested_cvs,
        "existing_files": existing_files,
        "processed_cvs": processed_cvs,
        "missing_files_count": len(missing_files),
        "gliner_errors_count": len(gliner_errors),

        "total_dataset_unique": total_dataset_unique,
        "total_dataset_occurrences": total_dataset_occurrences,
        "total_gliner_unique": total_gliner_unique,
        "total_gliner_occurrences": total_gliner_occurrences,
        "total_regex_unique": total_regex_unique,
        "total_regex_occurrences": total_regex_entity_occurrences,
        "total_intersection": total_intersection,
        "dataset_only_count": len(dataset_only_entities),
        "gliner_only_count": len(gliner_only_entities),

        "intersection_percentage": intersection_percentage,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "missing_files": missing_files,
        "gliner_errors": gliner_errors,
    }

    print("\n" + "=" * 80)
    print("DATASET ENTITIES vs GLINER (SKILLS + JOB TITLES) + REGEX LANGUAGES")
    print("=" * 80)
    print(f"Unique dataset entities                             : {total_dataset_unique}")
    print(f"Total dataset entity occurrences                    : {total_dataset_entity_occurrences}")
    print(f"Unique GLiNER entities + regex languages            : {total_gliner_unique}")
    print(f"Total GLiNER entity occurrences + regex languages   : {total_gliner_entity_occurrences}")
    print(f"Unique regex language-only entities                 : {total_regex_unique}")
    print(f"Total regex language-only occurrences               : {total_regex_entity_occurrences}")
    print(f"Matched entity occurrences (including overrides)    : {total_intersection}")
    print(f"Entities only in dataset                            : {len(dataset_only_entities)}")
    print(f"Entities only in GLiNER + regex languages           : {len(gliner_only_entities)}")
    print("-" * 80)
    print(f"Intersection percentage                             : {intersection_percentage:.2f}%")
    print(f"Precision                                           : {precision:.2f}%")
    print(f"Recall                                              : {recall:.2f}%")
    print(f"F1 score                                            : {f1:.2f}%")
    print("=" * 80)

    if missing_files:
        print(f"Missing files: {len(missing_files)}")

    if gliner_errors:
        print(f"GLiNER / regex errors: {len(gliner_errors)}")

    if summary_output_file:
        save_summary_report(summary_output_file, summary_data)

    return {
        "requested_cvs": requested_cvs,
        "existing_files": existing_files,
        "processed_cvs": processed_cvs,
        "summary_output_file": summary_output_file,
        "dataset_entities": dataset_unique_entities,
        "gliner_entities": gliner_unique_entities,
        "regex_entities": regex_unique_entities,
        "similar_pairs": [
            f"{dataset_entity}  <->  {gliner_entity}  [score={score:.2f}; match_type={match_type}]"
            for dataset_entity, gliner_entity, score, match_type in similar_info
        ],
        "dataset_only_entities": dataset_only_entities,
        "gliner_only_entities": gliner_only_entities,
        "dataset_counts": dataset_counts,
        "gliner_counts": gliner_counts,
        "regex_counts": regex_counts,
        "dataset_only_counts": dataset_only_counts,
        "gliner_only_counts": gliner_only_counts,
        "total_dataset_entity_occurrences": total_dataset_entity_occurrences,
        "total_gliner_entity_occurrences": total_gliner_entity_occurrences,
        "total_regex_entity_occurrences": total_regex_entity_occurrences,
        "missing_files": missing_files,
        "gliner_errors": gliner_errors,
        "intersection_percentage": intersection_percentage,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ----------------------------------------------------------
# Stopwords and truested words
# ----------------------------------------------------------

stopwords_exatas = [
    # usual generic trash of cvs
    "person", "first name", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "address", "house", "home",
    "academic", "academic background", "background","education", "experience", "hobbies","profile", "summary", "objective", "career objective","contact", "contacts", "personal details",
    "skills summary", "technical skills", "core competencies","work experience", "professional experience", "experience summary","education details", "academic details","Certification",
    "projects", "certifications", "achievements", "awards","activities", "interests", "languages", "references","competency", "competence", "area of expertise",
    "area of specialization", "professional skill","c1", "c2", "b1", "b2", "a1", "a2","first certificate","nationality","qualification","qualifications","diploma",
    "curriculum","religion","gender","phone","skype","reference","references","personality","duration","level","levels","members","team members", "achieved","achieve", "club",

    # gereric trash geral
    "TARGET", "text", "well", "and", "is", "are", "male","female", "can", "need", "like", "true", "false","meet", "personal","Computer","Digital","Computers",
    "achieve","reach","focused","written","accurate","results","value","needs","issues","history", "it", "per", "life", "lite", "dead",
    "qualification","qualifications","phone","skype","reference","references","duration","members","personality", "family", "familiar",
    "COM","work","information","knowledge","board","ltd","tech","Company","Technical","Organization","Passport", "use",

    # just verbs without meaning when alone
    "maintaining","implementing","updating","coordinating", "learning",
    "managing","processing","resolving","estimating","evaluation",
    "execution","exposure","opportunities","organisation",

    # just words withou meaning when alone
    "college","industrial","residential","commercial","trainings","permanent","growth", "images", "legal",
    "corporate","global","government","fields","value","cost", "river", "ocean", "hospital","closing",
    "processes","sports","travelling","water","wind", "dirt", "energy", "space", "air", "fire", "solar",

    #Countries and capitals
    "Country","capital","Countries","capitals","city","cities","UK","U.K", "U.K.","USA","U.S.A","U.S.A.","US",
    'Afghanistan', 'Kabul', 'Aland Islands', 'Mariehamn', 'Albania', 'Tirana', 'Algeria', 'Algiers', 'American Samoa', 'Pago Pago', 'Andorra', 'Andorra la Vella', 'Angola', 'Luanda', 'Anguilla', 'The Valley', 'Antarctica', '', 'Antigua And Barbuda', "St. John's", 'Argentina', 'Buenos Aires', 'Armenia', 'Yerevan', 'Aruba', 'Oranjestad', 'Australia', 'Canberra', 'Austria', 'Vienna', 'Azerbaijan', 'Baku', 'Bahamas The', 'Nassau', 'Bahrain', 'Manama', 'Bangladesh', 'Dhaka', 'Barbados', 'Bridgetown', 'Belarus', 'Minsk', 'Belgium', 'Brussels', 'Belize', 'Belmopan', 'Benin', 'Porto-Novo', 'Bermuda', 'Hamilton', 'Bhutan', 'Thimphu', 'Bolivia', 'Sucre', 'Bosnia and Herzegovina', 'Sarajevo', 'Botswana', 'Gaborone', 'Bouvet Island', '', 'Brazil', 'Brasilia', 'British Indian Ocean Territory', 'Diego Garcia', 'Brunei', 'Bandar Seri Begawan', 'Bulgaria', 'Sofia', 'Burkina Faso', 'Ouagadougou', 'Burundi', 'Bujumbura', 'Cambodia', 'Phnom Penh', 'Cameroon', 'Yaounde', 'Canada', 'Ottawa', 'Cape Verde', 'Praia', 'Cayman Islands', 'George Town', 'Central African Republic', 'Bangui', 'Chad', "N'Djamena", 'Chile', 'Santiago', 'China', 'Beijing', 'Christmas Island', 'Flying Fish Cove', 'Cocos (Keeling) Islands', 'West Island', 'Colombia', 'Bogota', 'Comoros', 'Moroni', 'Congo', 'Brazzaville', 'Congo The Democratic Republic Of The', 'Kinshasa', 'Cook Islands', 'Avarua', 'Costa Rica', 'San Jose', "Cote D'Ivoire (Ivory Coast)", 'Yamoussoukro', 'Croatia', 'Zagreb', 'Cuba', 'Havana', 'Cyprus', 'Nicosia', 'Czech Republic', 'Prague', 'Denmark', 'Copenhagen', 'Djibouti', 'Djibouti', 'Dominica', 'Roseau', 'Dominican Republic', 'Santo Domingo', 'East Timor', 'Dili', 'Ecuador', 'Quito', 'Egypt', 'Cairo', 'El Salvador', 'San Salvador', 'Equatorial Guinea', 'Malabo', 'Eritrea', 'Asmara', 'Estonia', 'Tallinn', 'Ethiopia', 'Addis Ababa', 'Falkland Islands', 'Stanley', 'Faroe Islands', 'Torshavn', 'Fiji Islands', 'Suva', 'Finland', 'Helsinki', 'France', 'Paris', 'French Guiana', 'Cayenne', 'French Polynesia', 'Papeete', 'French Southern Territories', 'Port-aux-Francais', 'Gabon', 'Libreville', 'Gambia The', 'Banjul', 'Georgia', 'Tbilisi', 'Germany', 'Berlin', 'Ghana', 'Accra', 'Gibraltar', 'Gibraltar', 'Greece', 'Athens', 'Greenland', 'Nuuk', 'Grenada', "St. George's", 'Guadeloupe', 'Basse-Terre', 'Guam', 'Hagatna', 'Guatemala', 'Guatemala City', 'Guernsey and Alderney', 'St Peter Port', 'Guinea', 'Conakry', 'Guinea-Bissau', 'Bissau', 'Guyana', 'Georgetown', 'Haiti', 'Port-au-Prince', 'Heard and McDonald Islands', '', 'Honduras', 'Tegucigalpa', 'Hong Kong S.A.R.', 'Hong Kong', 'Hungary', 'Budapest', 'Iceland', 'Reykjavik', 'India', 'New Delhi', 'Indonesia', 'Jakarta', 'Iran', 'Tehran', 'Iraq', 'Baghdad', 'Ireland', 'Dublin', 'Israel', 'Jerusalem', 'Italy', 'Rome', 'Jamaica', 'Kingston', 'Japan', 'Tokyo', 'Jersey', 'Saint Helier', 'Jordan', 'Amman', 'Kazakhstan', 'Astana', 'Kenya', 'Nairobi', 'Kiribati', 'Tarawa', 'Korea North', 'Pyongyang', 'Korea South', 'Seoul', 'Kuwait', 'Kuwait City', 'Kyrgyzstan', 'Bishkek', 'Laos', 'Vientiane', 'Latvia', 'Riga', 'Lebanon', 'Beirut', 'Lesotho', 'Maseru', 'Liberia', 'Monrovia', 'Libya', 'Tripolis', 'Liechtenstein', 'Vaduz', 'Lithuania', 'Vilnius', 'Luxembourg', 'Luxembourg', 'Macau S.A.R.', 'Macao', 'Macedonia', 'Skopje', 'Madagascar', 'Antananarivo', 'Malawi', 'Lilongwe', 'Malaysia', 'Kuala Lumpur', 'Maldives', 'Male', 'Mali', 'Bamako', 'Malta', 'Valletta', 'Man (Isle of)', 'Douglas, Isle of Man', 'Marshall Islands', 'Majuro', 'Martinique', 'Fort-de-France', 'Mauritania', 'Nouakchott', 'Mauritius', 'Port Louis', 'Mayotte', 'Mamoudzou', 'Mexico', 'Mexico City', 'Micronesia', 'Palikir', 'Moldova', 'Chisinau', 'Monaco', 'Monaco', 'Mongolia', 'Ulan Bator', 'Montenegro', 'Podgorica', 'Montserrat', 'Plymouth', 'Morocco', 'Rabat', 'Mozambique', 'Maputo', 'Myanmar', 'Nay Pyi Taw', 'Namibia', 'Windhoek', 'Nauru', 'Yaren', 'Nepal', 'Kathmandu', 'Netherlands Antilles', '', 'Netherlands The', 'Amsterdam', 'New Caledonia', 'Noumea', 'New Zealand', 'Wellington', 'Nicaragua', 'Managua', 'Niger', 'Niamey', 'Nigeria', 'Abuja', 'Niue', 'Alofi', 'Norfolk Island', 'Kingston', 'Northern Mariana Islands', 'Saipan', 'Norway', 'Oslo', 'Oman', 'Muscat', 'Pakistan', 'Islamabad', 'Palau', 'Melekeok', 'Palestinian Territory Occupied', 'East Jerusalem', 'Panama', 'Panama City', 'Papua new Guinea', 'Port Moresby', 'Paraguay', 'Asuncion', 'Peru', 'Lima', 'Philippines', 'Manila', 'Pitcairn Island', 'Adamstown', 'Poland', 'Warsaw', 'Portugal', 'Lisbon', 'Puerto Rico', 'San Juan', 'Qatar', 'Doha', 'Reunion', 'Saint-Denis', 'Romania', 'Bucharest', 'Russia', 'Moscow', 'Rwanda', 'Kigali', 'Saint Helena', 'Jamestown', 'Saint Kitts And Nevis', 'Basseterre', 'Saint Lucia', 'Castries', 'Saint Pierre and Miquelon', 'Saint-Pierre', 'Saint Vincent And The Grenadines', 'Kingstown', 'Saint-Barthelemy', 'Gustavia', 'Saint-Martin (French part)', 'Marigot', 'Samoa', 'Apia', 'San Marino', 'San Marino', 'Sao Tome and Principe', 'Sao Tome', 'Saudi Arabia', 'Riyadh', 'Senegal', 'Dakar', 'Serbia', 'Belgrade', 'Seychelles', 'Victoria', 'Sierra Leone', 'Freetown', 'Singapore', 'Singapur', 'Slovakia', 'Bratislava', 'Slovenia', 'Ljubljana', 'Solomon Islands', 'Honiara', 'Somalia', 'Mogadishu', 'South Africa', 'Pretoria', 'South Georgia', 'Grytviken', 'South Sudan', 'Juba', 'Spain', 'Madrid', 'Sri Lanka', 'Colombo', 'Sudan', 'Khartoum', 'Suriname', 'Paramaribo', 'Svalbard And Jan Mayen Islands', 'Longyearbyen', 'Swaziland', 'Mbabane', 'Sweden', 'Stockholm', 'Switzerland', 'Berne', 'Syria', 'Damascus', 'Taiwan', 'Taipei', 'Tajikistan', 'Dushanbe', 'Tanzania', 'Dodoma', 'Thailand', 'Bangkok', 'Togo', 'Lome', 'Tokelau', '', 'Tonga', "Nuku'alofa", 'Trinidad And Tobago', 'Port of Spain', 'Tunisia', 'Tunis', 'Turkey', 'Ankara', 'Turkmenistan', 'Ashgabat', 'Turks And Caicos Islands', 'Cockburn Town', 'Tuvalu', 'Funafuti', 'Uganda', 'Kampala', 'Ukraine', 'Kiev', 'United Arab Emirates', 'Abu Dhabi', 'United Kingdom', 'London', 'United States', 'Washington', 'United States Minor Outlying Islands', '', 'Uruguay', 'Montevideo', 'Uzbekistan', 'Tashkent', 'Vanuatu', 'Port Vila', 'Vatican City State (Holy See)', 'Vatican City', 'Venezuela', 'Caracas', 'Vietnam', 'Hanoi', 'British Virgin Islands', 'Road Town', 'U.S. Virgin Islands', 'Charlotte Amalie', 'USA Virgin Islands', 'Charlotte Amalie', 'Wallis And Futuna Islands', 'Mata Utu', 'Western Sahara', 'El-Aaiun', 'Yemen', 'Sanaa', 'Zambia', 'Lusaka', 'Zimbabwe', 'Harare',

    #USA Locations
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Jacksonville", "Fort Worth", "San Jose", "Austin", "Charlotte", "Columbus","Indianapolis", "San Francisco", "Seattle", "Denver", "Oklahoma City", "Nashville", "El Paso", "Washington D.C.", "Las Vegas", "Boston", "Portland", "Louisville", "Memphis", "Detroit", "Albuquerque", "Baltimore", "Milwaukee", "Fresno", "Sacramento", "Mesa", "Kansas City", "Atlanta", "Colorado Springs", "Raleigh", "Omaha", "Miami", "Virginia Beach", "Long Beach", "Oakland", "Minneapolis", "Bakersfield", "Tulsa", "Tampa", "Aurora",

    #Portugal Locations
    'Porto', 'Setubal', "Setúbal", "Castelo Branco",


    #MAIS
    "Mobile",
    "Software",
    "System",
    "Team",
    "Time",
    "Training",
    "Environment",
    "Report",
    "Music",
    "Professional",
    "USE",
    "Department",
    "MS",
    "Responsibilities",
    "Control",
    "Development",
    "Equipment",
    "Interpersonal",
    "Process",
    "Services",
    "Support",
    "XP",
    "Installation",
    "Maintain",
    "Maintenance",
    "Material",
    "Mechanical",
    "Organizational",
    "Problems",
    "Quality",
    "Reports",
    "Requirements",
    "solutions",
    "Distribution",
    "Panel",
    "Relationship",
    "Testing",
    "Tools",
    "Type",
    "Cable",
    "Goals",
    "Graduate",
    "Listening",
    "Monitor",
    "Oil",
    "Parts",
    "Production",
    "Progress",
    "Specifications",
    "Test",
    "age",
    "Challenging",
    "Commerce",
    "Cooling",
    "delivery",
    "Health",
    "Layout",
    "Life",
    "Load",
    "Make",
    "Operations",
    "Planning",
    "Proficiency",
    "Pumps",
    "Rapidly",
    "Repair",
    "RISK","Schedule","SEC","Star",
    "State",
    "Stock",
    "Tasks",
    "Transmission",
    "Treatment",
    "UPS",
    "Area of interest",
    "vista",
    "career",
    "Technical Qualification",
    "civil",
    "dir",
    "Job Responsibilities",
    "Married",
    "challenging position",
    "Executive",
    "Trainee",
    "abilities",
    "cricket",
    "Listening music",
    "QTP",
    "Basics",
    "e-mail",
    "potential",
    "accounts",
    "BOQ",
    "Areas of Interest",
    "DWG",
    "Listening to music",
    "craft persons",
    "job satisfaction",
    "laborers",
    "Occupation",
    "PCRs",
    "tenants",
    "Vogue",
    "Adept",
    "expert",
    "Flaunt",
    "Speak",
    "Sr.",
    "swimming",
    "Glamour",
    "HOD",
    "Junior resident",
    "Linguistic Proficiency",
    "Playing cricket",
    "Write",
    "Askim Online English Tutorial",
    "Badminton",
    "Declaration",
    "Elle",
    "HAP",
    "member",
    "Personal Strength",
    "post",
    "Signature",
    "Singing",
    "Single",
    "Supervisors",
    "Turkish students",
    "American Idol",
    "attained maximum profit",
    "CEO",
    "CG Lux",
    "Corel draw",
    "dancing",
    "DCA",
    "erection",
    "Field of Interest",
    "Firefighting system",
    "First Class",
    "General",
    "incumbent",
    "KEIL",
    "Major",
    "MEP field",
    "NDT",
    "O & M",
    "Plan",
    "profession",
    "R & D",
    "Runway",
    "Site Incharge",
    "SLD",
    "Specialist",
    "talents",
    "Topic",
    "Training & Development",
    "VHDL",
    "ACB",
    "Apprentice",
    "Basic Core",
    "BasicKnowledgeofComputer",
    "BOSS",
    "C & I",
    "casting",
    "challenging career",
    "Challenging job",
    "Dealers",
    "Event organizer",
    "Graduation",
    "hair",
    "Incharge",
    "ISI testing of MCBs",
    "key player",
    "KEY Responsibilities",
    "key role",
    "Lead",
    "Matriculation",
    "MIG",
    "patients",
    "PDMS",
    "Personal Information",
    "personal trait",
    "Personality Traits",
    "Phillips Driver",
    "Position Held",
    "Predictive",
    "Preparation of Bills of documents",
    "Preparation of programme of works",
    "Preparation of Tender documents",
    "Preparation of variation orders",
    "Professional environment",
    "Resident",
    "student",
    "students",
    "Subject",
    "symbiotic relationship",
    "Teen Vogue",
    "Testing & Quality Control Engineer Representative",
    "Tiles Mason",
    "value addition",
    "VFD",
    "Work Order",
    "workers",
    "7QC tools",
    "Accounting Package",
    "Aim",
    "ARC",
    "Architectural",
    "as a Graduate Engineer",
    "as a safety officer",
    "asst",
    "attitude",
    "Austin Powers",
    "Basics of launguages",
    "Beginner",
    "Bluemix",
    "Brake test",
    "Branch",
    "calculation",
    "challenging and responsible position",
    "Checkpoint",
    "Communicator",
    "Confined Space",
    "Cost ascertainment",
    "Dept.","DTP","ECE","EFA","Effective communicator",
    "Employee",
    "Engg",
    "EViews",
    "exel",
    "Familiarity",
    "field",
    "Fundamentals",
    "Global Warming",
    "Graduate Apprentice Trainee",
    "high-level position",
    "HSE",
    "in charge",
    "in depth knowledge",
    "In Style",
    "Insta llation of Raised floor and Reflected ceilings",
    "install",
    "Interest",
    "interface",
    "ISTP",
    "key position",
    "Key Result Areas",
    "Knowledge in I nstallation of precast hollow core roof slabs",
    "LA Confidential",
    "ledger",
    "level of study",
    "Life member",
    "Manual",
    "Marie Claire",
    "MCCB",
    "MDB",
    "Method",
    "Micrometer",
    "MIGO",
    "My Role",
    "Obtaining DEWA engineer",
    "P&M",
    "P.Point",
    "packing issuer",
    "PCC",
    "permit to work",
    "Personal Interests",
    "Personal Qualities",
    "Reading novels","RTD",
    "S oil density testing","SMDB","Software Skils","soun d understanding","Specialisation",
    "ss7","staffs","Structural","suitable position","suitable post","Surfing","Technical Abilties",
    "technical inputs","technical issues","Technical Qualifications","TIG","towards life",
    "Unmarried","Europe","Power",
    "Client","Ensure","People",
    "Reading",
    "Records",
    "Clients",
    "Advertising",
    "Box",
    "build",
    "Location",
    "Meetings",
    "Completion",
    "Duties",
    "Pressure",
    "Dynamic",
    "Detail",
    "Develop",
    "Employment",
    "Regular",
    "Staff",
    "Success",
    "Supply",
    "Park",
    "FAIR",
    "Read",
    "Commercials",
    "Flow",
    "Good communication",
    "Organized",
    "Products",
    "Lighting",
    "Scope",
    "Continuous",
    "Manpower",
    "VisIT",
    "MINI",
    "Schedules",
    "Size",
    "Workshop",
    "List",
    "Medical",
    "Salary",
    "Change",
    "Changes",
    "Guide",
    "Heat",
    "Deadlines",
    "FAST",
    "MAN",
    "Termination",
    "Approval",
    "Green",
    "Studies",
    "Leading",
    "Trends",
    "Internship",
    "pH",
    "PO",
    "Secure",
    "Carrier",
    "Employees",
    "Entertainment",
    "MBA",
    "Videos",
    "Aggregate",
    "Video",
    "Advance",
    "Customers",
    "Events",
    "Packages",
    "Files",
    "Film",
    "Identifying",
    "Card",
    "Balance",
    "Collection",
    "RED",
    "SAFe",
    "Strengths",
    "Payments",
    "Trust",
    "Wall",
    "CAR",
    "Practice",
    "Ray",
    "requests",
    "Arts",
    "Grant",
    "HV",
    "Insurance",
    "Invoices",
    "Movies",
    "Queries",
    "Strength",
    "Excellence",
    "Physical",
    "Prevention",
    "Reconciliation",
    "Telephone",
    "Complaints",
    "Credit",
    "Culture",
    "Exam",
    "HP",
    "Loss",
    "RedBook",
    "Response",
    "Sheets",
    "STORE",
    "Actions",
    "Expansion",
    "Initiative",
    "Jobs",
    "Roles",
    "Studying",
    "Agencies",
    "Cement",
    "Delta",
    "Impact",
    "Julia",
    "Participate",
    "Replacement",
    "Media",
    "Speed",
    "Talk",
    "Clean",
    "Comments",
    "Competitions",
    "Permits",
    "Presentations",
    "Publications",
    "Rules",
    "Waste",
    "Weight",
    "Action",
    "Bridges",
    "Conferences",
    "Defects",
    "Diverse",
    "ETA",
    "Identification",
    "Induction",
    "Notes",
    "Options",
    "Retail",
    "Reviews",
    "Temperature",
    "Typing",
    "Vehicles",
    "Voice",
    "ADAM",
    "Community",
    "Driven",
    "Exchange",
    "Facts",
    "Feedback",
    "Filing",
    "Inc",
    "Instructions",
    "IP",
    "Journals",
    "Kingdom",
    "Maria",
    "Monthly reports",
    "PG",
    "SHARE",
    "SOURCE",
    "Steam",
    "Travel",
    "vision",
    "ABB",
    "ADAMS",
    "Adapt",
    "Children",
    "COLA",
    "Consultants",
    "Consumption",
    "Crystal",
    "Football",
    "Innovation",
    "Play",
    "Radio",
    "RAISE",
    "Relationships",
    "Search",
    "Seminars",
    "Stream",
    "Updates",
    "Atmosphere",
    "Black",
    "Correspondence",
    "Hospitals",
    "Motivation",
    "Traveling",
    "5S",
    "Accidents",
    "Arena",
    "CE",
    "Challenge",
    "Incident",
    "LE",
    "Manuals",
    "Papers",
    "Cold",
    "Daily Reports",
    "Express",
    "Hand",
    "Interaction",


    "Ability", "Responsibility", "Accuracy", "Capacity", "Efficiency",
    "Improvement", "Performance", "Utilization",
    "Leadership", "Communication", "Communicate",
    "Writing", "Presentation",
    "Innovative", "Analytical",
    "Senior","HR", "QC","QA",
    "Professional Qualification",
    "Compliance",
    "Practices", "Procedures", "Procedure",
    "Policies", "Regulations",
    "Review", "Verification","Codes",
    "Preparation","Standards", "Policy", "Methods",
    "Handling","Technological", "Functional",
    "Functions",
    "Structure",
    "Resource", "Resources",
    "Budget",
    "Service",
    "Customer",
    "Plans",
    "Survey",
    "Reporting",
    "Safety",
    "Security",
    "Quality",
    "KV",
]

stopwords_parciais = [
    # more genric trash
    "certificate","academic","education","experience","profile","summary",
    "objective","contact","project","projects","language","languages","competenc","skill",
    "name", "email", "gmail", "@", ".com", "certificate", "background", "degree",
    "hobbies", "highschool", "high school","secondary school", "bachelor", "bachelors",
    "licenciatura", "curriculum vitae", "resume","work experience", "professional experience",
    "career objective", "personal details","contact details", "technical skills",
    "core competencies", "undergraduate","graduação", "licenciado", "master", "masters",
    "postgraduate", "pós-graduação", "mestrado","mestre", "mestrado integrado",
    "doctorate", "doctoral", "doutoramento", "doutorado","phd", "ph.d", "dphil", "d.phil",

    #just why
    "curriculum","nationality","marital","religion","gender","gay","LGBT", "mother", "father", "brother", "sister",
]

trusted_words=["c++","poka yoke","poka-yoke","poka -yoke", "poka-yoke", "deep learning","deeplearning","creative",
    "confident","positive thinking","responsible","sincere","good listener","hard worker","risk assessment","site execution","team worker",
    "disciplined","net surfing","ambitious","fire fighting","firefighting","co-ordination","conduiting","confident and determined",
    "coordination","confident", "determined","IT proficiency","team member","leader","good leader","team leader",
    "internet browsing","arthroplasty","loop checking","suggestive selling","plastering","maintenance team","active listener","lathe",
    "leadership quality","handling quick books", "making friends", "self-confident","multi-tasking","machine maintenance","manage",
    "keen observer","server side technologies", "IT forte","duct sizing", "thermal testing", "Electrical internal & External Electrification",
    "TRT testing","driving","good communication","honest","artistic","banking","brand maintenance","teaching","housing","respect","availability",
    "team leader","coordinate","smart","drive","friendly","carrying","tenure","filling","dedicated","positive attitude","power distribution",
    "efficient","motivated","learner","trading","positive attitude","preventive","preventive & maintenance","critical thinker","cross-team","department head",
    "civil helper","civil maintenance","breakdown maintenance","supervise","analyze","computer science","marketing","finance","networking","driving license",
    "flexible", "Fixing", "excellent communication","organize", "supervising","games","sciences","reliability",


    # First 500
    'account reconciliation', 'accounts receivable', 'AutoCAD', 'AVR Studio', 'BMS interface', 'bookkeeping', 'C', 'C++', 'cash book', 'close loop checking', 
    'CMD software', 'CNC', 'confined space entry', 'CoralDRAW', 'Coral Video Studio', 'design & layout', 'detailed estimation', 'DGPS', 'drafting software', 
    'earth resistance testing', 'electrical panel maintenance', 'erection & commissioning', 'fire prevention & fire fighting', 'First Aid', 'fixed assets', 
    'front end', 'function test', 'GD&T', 'H2S awareness', 'hazard identification', 'heat load calculation', 'high pot testing', 'HV testing', 'hydro testing', 
    'Induction Motors', 'insulation resistance test', 'isometric drawing', 'JSA', 'keyword optimization', 'lighting layout', 'Load Runner', 'lumen testing',
    'maintenance', 'manpower planning', 'manual handling', 'microcontroller', 'microprocessor', 'Microsoft Office', 'operation & maintenance', 
    'open loop checking', 'password management', 'permit to work system', 'site supervision', 'testing', 'tool box talk', 'web/WWW', 'Win XP', 
    'Win8', 'WordPress plugin', 'working at height', 'Xilinx',

    # 500 - 1000
    'PFD', 'PH meter', 'pressure gauge', 'pressure testing', 'pressure transmitter', 'Pro/E', 'ProE 2000i', 'pump head calculation', 'PVSyst', 'R software', 
    'relays', 'Sand test', 'scaffold safety', 'scaffolding', 'smoke testing', 'SMTP', 'specific gravity', 'STAAD Pro', 'star delta', 'stock ledger', 
    'subdivision distribution board', 'super tension cable', 'Tally', 'Tally ERP', 'TeamViewer', 'telephone systems', 'testing & commissioning', 'total station', 
    'transmission & distribution', 'ultrasonic testing', 'web design', 'Windows', 'WinCC', 'working at height', '2D drawing', '3-phase power systems', '5S', '7QC tools', 
    'Arc welding', '8051 microcontroller', 'AutoCAD', 'CAD tools', 'cash flow analysis', 'CCTV/IP camera setup', 'control systems', 'electrical design', 'estimation', 
    'HVAC systems', 'inspection', 'instrumentation', 'maintenance', 'mechanical work', 'microcontroller', 'panel modifications', 'preventive maintenance', 
    'quality control', 'scaffolding safety', 'testing', 'valves', 'welding',

    # 1000 - 1500
    'ARPA', 'artificial intelligence', 'asset management', 'audit', 'AutoCAD', 'AutoCAD 2D', 'AutoPLANT', 'AviCAD', 'AWG', 'AWT', 'balance sheet', 'bank reconciliation', 
    'bar bending schedule', 'Barracuda firewall', 'basic firefighting', 'BMS', 'BMS & HVAC basics', 'board testing', 'bookkeeping', 'books of accounts', 'boundary survey', 
    'breakdown maintenance', 'bug tracking', 'Bugzilla', 'building construction', 'building wiring', 'business analysis', 'business development', 'C', 'C++', 'CAD', 
    'CADWorx', 'CAE software', 'Caesar II', 'calculation', 'cable laying', 'cable sizing', 'cable testing', 'CAD drafting', 'cash flow analysis', 'CCTV', 'control systems', 
    'CPR', 'CRM', 'data analysis', 'electrical engineering', 'estimation', 'firefighting', 'HVAC', 'inspection', 'instrumentation', 'IT systems', 'maintenance', 
    'mechanical engineering', 'PLC', 'SCADA', 'testing', 'welding',

    # 1500 - 2000
    'Capacitance measurement', 'capital asset identification', 'carbon dioxide detection', 'card testing', 'cardio pulmonary rehabilitation', 'cash book', 'cash handling', 
    'CATIA V5', 'CB sizing calculation', 'centrifugal pumps', 'centroid calculation', 'change management', 'checkpoint firewalls', 'circuit design', 'circuit theory', 
    'Citect SCADA', 'civil drafting', 'civil engineering', 'claim processing', 'client billing', 'client handling', 'CNC machines', 'CNC programming', 'cold calling', 
    'collection of payments', 'commercial evaluation', 'commissioning', 'communication skills', 'component testing', 'compressor maintenance', 'computer literacy', 
    'concrete batching', 'concrete casting', 'concrete mix design', 'conductivity measurement', 'confined space entry', 'construction management', 'construction quality management', 
    'control systems', 'control panels', 'control circuits', 'control logix', 'coordination', 'corporate finance', 'cost accounting', 'cost control', 'cost reduction', 'CPR', 
    'credit appraisal', 'credit control', 'Creo/Pro-E', 'crimping', 'critical task analysis',

    # 2000 - 2500
    'cross cut adhesion testing', 'cross selling', 'CRUD', 'cryptanalysis', 'CT analyzer', 'CT/PT sizing', 'current transformer sizing', 'Curve Tracer', 
    'customer relationship management', 'Cyberoam firewall', 'Cygwin', 'data validation', 'DC sheath test', 'DC winding resistance', 'DCS FAT', 'debridement', 
    'debtors & creditors', 'design & development', 'design drafting', 'detailed drawings', 'develop test cases', 'device net', 'dew point measurement', 
    'DG set maintenance', 'DG sizing calculation', 'DGPS operations', 'diagnosing faults', 'DIALux', 'digital marketing', 'direct torque control', 'disaster management', 
    'dispatch management', 'distribution planning', 'document control', 'documentation', 'DOL starter', 'Dropbox', 'DSP', 'DTMF decoder', 'duct design', 'duct leak test', 
    'dumpy level', 'e-commerce', 'e-mail', 'E-plan', 'e-recruitment', 'E-survey', 'Eagle Layout Editor', 'earth resistance testing', 'earthworks', 'earthquake resistant design', 
    'ECAD', 'EBITDA', 'EHS awareness', 'electrical control circuits', 'electrical distribution', 'electrical installation', 'electrical load calculation', 'electrical maintenance', 
    'electrical systems maintenance', 'electromagnetism', 'electronics design', 'embedded C', 'embedded systems', 'emergency management',

    # 2500 - 3000
    'emulation', 'cryptography', 'energy management', 'EPABX systems', 'equipment layout', 'equipment selection', 'erection & commissioning', 'ERP modules', 
    'error handling', 'Ethernet IP', 'ETL', 'evacuation planning', 'excavation', 'Excel', 'Exchange Server', 'export/import', 'extraction & reporting', 'fabrication', 
    'fault current analysis', 'FAT analysis', 'FCAW welding', 'feedback loops', 'feeder maintenance', 'field calibration', 'field instrumentation', 'FileZilla', 
    'filter press operation', 'finalization of accounts', 'financial analysis', 'fire detection systems', 'fire fighting design', 'fire prevention', 'firewall', 
    'firmware upgrade', 'first aid', 'fishbone analysis', 'fixed asset management', 'flow measurement', 'Fluke instruments', 'foundation work', 'front office management', 
    'functionality testing', 'fund flow analysis', 'gas cutting', 'gas welding', 'gate valves', 'GD&T', 'general ledger', 'generating leads', 'geotechnical analysis', 
    'GMAW welding', 'GNU', 'Google Maps', 'gloss measurement',

    # 3000 - 3500
    'AutoCAD', 'PlanSwift', 'GRN (Goods Receipt Note)', 'graphic software', 'grouting', 'GSM', 'GST', 'heat load calculation', 'heat transfer', 'heat treatment', 
    'hardness testing', 'hardware testing', 'hash tables', 'hazard management', 'HAZOP', 'HAP software', 'HVAC load calculation', 'HVAC systems', 'handset repair', 
    'HMI (human machine interface)', 'humidity measurement', 'hydraulic systems', 'pneumatic systems', 'hydro testing', 'HPLC', 'HR management', 'HSE management', 
    'HTML/HTTPS', 'hypercard', 'IBS testing', 'ICT support', 'identifying hazards', 'impact testing', 'import/export', 'incident management', 'incoming inspection', 
    'industrial maintenance', 'inergen systems', 'information systems', 'inspection', 'installation & commissioning', 'instrumentation', 'insulation resistance testing', 
    'integrated management systems', 'interpersonal skills', 'inventory control', 'in-process quality control', 'innovation', 'process improvement',

    # 3500 - 4000
    'international marketing', 'internet concepts', 'internet programming', 'inventory management', 'inverter systems', 'iptables', 'IPTV', 'isometric drawing', 
    'IT desktop support', 'IT maintenance', 'ITIL', 'Jasper Reports', 'JQuery Mobile', 'k-means clustering', 'Kaizen', 'Karl Fischer titration', 'Key Account Management', 
    'KNX systems', 'LAN networking', 'land surveying', 'lathe operations', 'layout design', 'leak testing', 'lean methodology', 'ledger management', 'legal documentation', 
    'letter of credit', 'level transmitter', 'lighting load calculation', 'liquid penetrant testing', 'load calculation', 'lockout tagout (LOTO)', 'logistics management', 
    'loop checking', 'lubrication systems', 'lux calculation', 'magnetic particle testing', 'mail server management', 'maintenance & troubleshooting', 'maintenance management', 
    'MIS reporting', 'motor maintenance', 'machine handling', 'machine maintenance',

    # 4000 - 4500
    'audit', 'makefile', 'MAMP', 'management accounting', 'management information systems', 'manual testing', 'MapInfo', 'market analysis', 'material handling', 'material planning', 
    'material take-off', 'Mathematica', 'Maximo', 'McQuay DuctSizer', 'mechanical design', 'MEP design', 'Mercury Quality Centre', 'Metasys', 'meter testing', 'microgrid', 
    'MicroStation', 'microbial testing', 'microcontroller 8051', 'microcontroller AT89C51', 'MicroLogix', 'microprocessor 8085', 'MIG welding', 'TIG welding', 'MikroTik', 
    'MIS audits', 'mixing and fluid testing', 'ModelSim', 'module testing', 'monitoring and inspection', 'MPLAB IDE', 'MPPT controller', 'MS Office', 'MS PowerPoint', 
    'MS Publisher', 'MS-DOS', 'VLOOKUP', 'MTBF', 'MTTR', 'multimedia systems', 'multivariable statistics', 'MVC', 'Myrinet', 'numerical relay testing', 'Navicat', 'Navisworks', 
    'NDT', 'network systems', 'NGT/NGR sizing', 'nitration', 'NNTP', 'non-destructive testing', 'normal and emergency lighting', 'motor maintenance', 'maintenance & repair', 
    'maintenance troubleshooting', 'maintenance planning', 'cash handling', 'ledger scrutiny', 'invoice processing', 'banking operations', 'record keeping', 'safety management', 
    'HSE management', 'site monitoring', 'installation & commissioning', 'inspection', 'mechanical works', 'electrical maintenance', 'cable laying', 'lubrication',

    # 4500 - 5000
    'Novec 1230 systems', 'numerical analysis', 'object oriented programming', 'object repository', 'object spy', 'OFC checking', 'OHSAS', 'oil handling', 'oil in water analysis', 
    'Omron HMI', 'OpenNMS', 'operating systems', 'operation & maintenance', 'Oracle 10g', 'Oracle 11g', 'OrCAD', 'order to cash', 'osCommerce', 'Outlook Express', 'overhead tank sizing', 
    'P&ID', 'P&ID mark-up', 'package AC systems', 'palletizing', 'Pareto analysis', 'particle count testing', 'password cracking', 'payroll management', 'PCB design', 'PCC & MCC panels', 
    'PCS7', 'PDF tools', 'Peachtree accounting', 'penetrant testing', 'pencil hardness testing', 'performance analysis', 'period close analysis', 'petroleum exploration', 'PF Sense firewall', 
    'phase sequencer', 'physical metallurgy', 'physiotherapy techniques', 'PIC microcontrollers', 'PID isometric drawings', 'pile foundation design', 'piping design', 'piping layout', 
    'pivot tables', 'planning & scheduling', 'plant downtime calculation', 'plasma cutting', 'plastering', 'PLC programming', 'plot plans', 'plumbing systems', 'point of sale (POS)', 
    'poka-yoke', 'polymer synthesis', 'POP3', 'poster design', 'postoperative pain management', 'power electronics', 'power plant management', 'power transmission',

    # 5000 - 5500
    'PPAP', 'PPC scheduling', 'practical accounting', 'pre-weld inspection', 'precast concrete', 'predictive maintenance', 'preventive maintenance', 'BOQ preparation', 
    'tender preparation', 'bill preparation', 'as-built drawings', 'bar bending schedules', 'Primavera P6', 'Pro/E', 'problem solving', 'process planning', 'process safety', 
    'process control', 'invoice processing', 'procurement', 'production planning', 'production management', 'quality control', 'quality assurance', 'quality systems', 'Profibus', 
    'profit & loss accounting', 'programming', 'proofreading', 'prototype development', 'Proteus', 'PScad', 'psychrometric chart', 'pump head calculation', 'PV Elite', 'PV systems', 
    'QC inspection', 'QA/QC', 'quantity surveying', 'QuickBooks', 'quotation management', 'radiographic testing', 'rainwater harvesting', 'RCCB', 'rebar fixing', 'reconciliation of accounts', 
    'recruitment', 'reinforcement checking', 'relationship management', 'Relux lighting software', 'repair & maintenance', 'report writing', 'research', 'resource management', 'risk management', 
    'root cause analysis', 'reading blueprints',

    # 5500 - 6000
    'risk assessment', 'rigging operations', 'rigging safety', 'rock testing', 'RSLogix', 'RSView32', 'rsync', 'RTOS', 'RTGS', 'safety management', 'sales planning', 'sales & marketing', 
    'sample preparation', 'sanity testing', 'SAP BI', 'SAP FI', 'SCADA', 'scaffolding', 'scaffolding safety', 'schedule preparation', 'scheme checking', 'Scilab', 'scope meter', 'screening candidates', 
    'scrutiny', 'sealing work', 'seaming machines', 'secondary surveys', 'selection & power calculation', 'self compacting concrete', 'selling techniques', 'server 2008', 'service & maintenance', 
    'sewage treatment plant', 'Sewercad', 'sheet piling', 'sheet metal fitting', 'shift planning', 'short circuit calculation', 'shutdown planning', 'Silverlight', 'site evaluation', 'site execution', 
    'site inspection', 'site mobilization', 'SketchUp', 'slab checking', 'sling psychrometer', 'slitting', 'smart grid', 'smart meters', 'SmartPlant 3D', 'SmartPlant P&ID', 'SMAW welding', 'SMPS', 'SMT', 
    'software development', 'software testing', 'software testing life cycle', 'soil impact test', 'solar energy', 'solar panel installation', 'solar thermal design', 'SolidWorks', 'Solid Edge', 
    'SonicWall firewall', 'SOP records', 'spatial indexing', 'speed control', 'spectrophotometer', 'spin coating', 'spinal surgery techniques', 'spot welding', 'spray nozzle systems', 
    'sprint burndown charts', 'statistical quality control', 'SQC methods',

    # 6000 - 6500
    'SRAM', 'SRS', 'STAAD Pro', 'stabilization of patients', 'stage-wise testing', 'stakeout method', 'stamp concrete', 'star-delta starter', 'static pressure calculation', 'static regain method', 
    'statutory audit', 'statutory reporting', 'steam blowing', 'steam turbine', 'steel erection', 'steel fixing', 'steganography', 'stock audit', 'stock management', 'stock taking', 
    'stress analysis software', 'structural modelling', 'student supervision', 'Studio 5000', 'subnetting', 'SunOS', 'supervision & control', 'supplier quality management', 'surface preparation', 
    'survey software', 'symmetric key cryptography', 'Systat', 'system integration', 'system testing', 'Tally ERP', 'team building', 'technical evaluation', 'technical support', 'telecalling', 
    'telecom systems handling', 'Telerik', 'temperature transmitter', 'temperature & RH testing', 'TEMS', 'tendon suturing', 'tensile testing', 'testing & commissioning', 'earth resistance testing', 
    'thermocouple', 'RTD', 'thermodynamics', 'thermal cycling', 'three-phase induction motor starter', 'theory of machines', 'tool box talks', 'topographic survey', 'torque testing', 
    'total quality management', 'total station', 'total sulfur analyzer',

    # 6500 - 7000
    'TIA Portal / Totally Integrated Automation', 'tower crane erection', 'tower erection', 'tower maintenance', 'traceability matrix', 'traffic safety', 'first aid', 'training coordination', 
    'training needs analysis', 'transformer sizing', 'transmission systems', 'traverse survey', 'trend and variance analysis', 'Trend Micro Antivirus', 'trial balance', 'troubleshooting', 
    'ultrasonic testing', 'UG NX', 'Unigraphics NX', 'unit testing', 'UPS systems', 'utility and plant maintenance', 'V-Ray', 'valve systems', 'variance analysis', 'vector network analysis', 
    'vendor management', 'VFD', 'visual and dimensional inspection', 'Visual Basic', 'Visual Studio', 'VLSI', 'VNC', 'voltage drop calculation', 'voltage regulation', 'VRML', 'VRF/VRV systems', 
    'WAMP', 'WAN', 'waste management', 'effluent management', 'waterproofing', 'WaterCAD', 'WBS planning', 'web testing', 'website testing', 'welding', 'cutting', 'well testing', 'white box testing', 
    'Wi-Fi / radio systems', 'WinBox', 'WinRAR', 'WinSCP', 'wire bonding', 'witness pressure testing', 'WMM checking', 'workshop management', 'WPS', 'WSUS', 'Xcode', 'Xilinx ISE', 'yield calculation', 'YOLO CNN',

    # EN
    "teacher", "professor", "manager", "analyst", "engineer", "developer",
    "consultant", "coordinator", "supervisor", "assistant", "technician",
    "driver", "lifeguard", "operator", "pilot", "scientist", "private security",
    "security officer", "security guard", "deliveryman", "forester","biochemist",
    "food scientist", "geneticist","farmer", "biologist", "horticulturalist",
    "inspector","beekeeper","ecologist","conservationist", "veterinarian",
    "zoologist", "groomer", "animal control officer", "attendant", "pet walker",
    "animal sitter", "breeder", "pathologist","ophthalmologist", "reporter",
    "actuary","clerk", "artist", "therapist", "barber", "beautician", "designer",
    "stylist","cashier", "coordinator", "advisor", "writer", "editor", "illustrator",
    "actor","singer","producer","web developer","architect","animator", "painter",
    "tutor", "headteacher", "counsellor", "nurse", "doctor", "personal trainer",
    "pharmacist","dentist", "orthodontist", "dietitian", "housekeeper", "porter",
    "tour guide", "supervisor", "director", "makeup artist","hairstylist","lecturer",
    "professional staff","engineer","intern","developer","assistant","captain",
    "analyzer","wardrobe stylist","chairman","accountant","fire fighter","sailor",
    "co-ordinator","coordinator","orthopaedics","plumber","account assistant",
    "quality control & testing department","quality control & testing engineer",
    "quantity survey assistant","computer operator & store department","assistant Engineer",
    "head of the department","house captain","orthopaedic surgeon","surgeon","technicians",
    "technician","secretary","site consultants","super admin","super-admin","admin",
    "referees","Manicurist","assistant manager", "hunter","Hairstylist & Makeup Artist",

    # First 500
    'Assistant General Manager', 'Business Development Executive', 'Cashier/Accountant', 
    'Civil Supervisor', 'Document Controller', 'Draughtsman', 'Duct Charge Hand', 'Duct Man', 
    'Electrical Supervisor', 'Founder', 'Graduate Engineer Trainee', 'HR Assistant', 'HR Executive', 
    'HSE Officer', 'Instructor', 'Junior Engineer', 'Junior Executive', 'Junior Specialist', 
    'Logistics/Store Executive', 'Locum Specialist', 'Maintenance Engineer', 'Manager MEP Services', 
    'Marketing Executive', 'Medical Doctor', 'Medical Officer', 'Medical Representative', 'MEP Head', 
    'MIS Executive', 'Office Executive', 'Road Surveyor Assistant', 'Site In Charge', 'Store Keeper', 
    'Technical Coordinator', 'Technical Manager', 'Volunteer Assistant Accountant', 'Worker',

    # 500 - 1000
    'AC Technician', 'Account & Finance Executive', 'Account Manager', 'Account Officer', 
    'Accounts Assistant', 'Admin Officer', 'Administration Officer', 'Advocate', 'Aluminium Fabricator', 
    'Assistant Production Manager', 'Credit Analyst', 'Electrical Design Engineer', 'Field Engineer', 
    'Field Survey Officer', 'HVAC Technician', 'IT Support Engineer', 'Maintenance Engineer', 'Manager', 
    'Production Engineer', 'Production Manager', 'Sales Executive', 'Salesman', 'Senior Executive', 
    'Shift Engineer', 'Site Manager', 'Specialist Intern', 'Store Executive', 'Teaching Assistant', 
    'Trainee Engineer', 'Transmission Engineer', 'Vice President', 'Web Designer', 'Web Designer & Developer',

    # 1000 - 1500
    'Account Coordinator', 'Accounts Assistant', 'Administration Assistant', 'Area Sales Manager', 
    'Article Clerk', 'Assistant Commandant', 'Assistant Lecturer', 'Assistant Manager', 'Assistant Site Engineer', 
    'Assistant Storekeeper', 'Audit Assistant', 'Auditor', 'Back Office Executive', 'Banking Training Manager', 
    'BI Consultant', 'Branch Engineer', 'Business Development Executive', 'CAD Draughtsman', 'Civil Engineer', 
    'Commercial Officer', 'Deputy Manager', 'Duty Manager', 'Electrical Engineer', 'Engineer', 'Field Assistant', 
    'Financial Manager', 'HSE Officer', 'HVAC Engineer', 'IT Engineer', 'Lecturer', 'Logistics Coordinator', 
    'Maintenance Engineer', 'Mechanical Engineer', 'Office Assistant', 'Portfolio Manager', 'Production Manager', 
    'Quality Engineer', 'RF Engineer', 'Sales Assistant', 'Scaffolding Coordinator', 'Scaffolding Specialist', 
    'Security Officer', 'Shift Engineer', 'Site Engineer', 'Station Manager', 'Store Manager', 'Technical Assistant', 
    'Technician', 'Telecommunication Engineer', 'Test Manager', 'Testing Engineer', 'Trainee Engineer', 'Web Designer',

    # 1500 - 2000
    'Accountant', 'Assistant Manager', 'Assistant Site Engineer', 'Cashier', 'Celebrity Makeup Artist', 'Chairman', 
    'Chief Accountant', 'Chief Engineer', 'Civil Draughtsman', 'Civil Engineer', 'Civil Foreman', 'Civil Quantity Surveyor', 
    'Civil Site Supervisor', 'Clerk', 'Client Executive', 'Commercial Executive', 'Commissioning Engineer', 'Company Secretary',  
    'Consultant Engineer', 'Consultant Orthopaedic Surgeon', 'Coordinator', 'Cost Accountant', 'County Administrator', 
    'Engineer', 'Manager', 'Production Manager', 'Sales Manager', 'Security Officer', 'Site Supervisor', 'Store Manager', 'Technician',

    # 2000 - 2500
    'Accounts Executive', 'Assistant Instruments Technician', 'Customer Relationship Executive', 'Dean of Students', 'Deputy Finance Manager', 
    'Deputy Manager', 'Deputy Team Leader', 'Design Consultant', 'Design Engineer', 'Desk Executive', 'Development Manager', 'Diesel Mechanic', 
    'Diploma Engineer', 'Diploma Engineer Trainee', 'Director', 'Distinguished Professor', 'Document Controller', 'Draftsman', 'Economist', 
    'Editor in Chief', 'Editorial Board Member', 'Electrical Engineer', 'Electrical Instrument Officer', 'Electrical Supervisor', 'Electrician', 
    'Emergency Medical Officer', 'Engineer', 'Estimation Engineer', 'Executive', 'Finance Manager', 'Instruments Technician', 'Manager', 
    'Orthopedic Surgeon', 'Piping Stress Engineer', 'Production Manager', 'Professor', 'Relation Ship Manager', 'Secretary', 'Service Engineer', 
    'Shift In-Charge', 'Site Engineer', 'Supervisor', 'Teaching Assistant', 'Technician', 'Team Leader',

    # 2500 - 3000
    'Entrepreneur', 'Enumerator', 'Engineer Trainee', 'Engineering Consultant', 'Executive Secretary', 'Excise Clerk', 'Exercise Grader', 
    'Facility Executive', 'Factory Manager', 'Faculty Member', 'Finance Executive', 'Fire and Security Officer', 'Fitter', 'Foreman', 
    'Foundation Doctor', 'Fund Administrator', 'General Accountant', 'General Manager', 'General Practitioner', 
    'Global Lead Knowledge Manager', 'Glims Administrator',

    # 3000 - 3500
    'Government Apprentice Trainee', 'Graduate Engineer', 'Guest Relation Officer', 'HSE Officer', 'Hair Stylist', 'Hand Surgeon', 
    'Head of Department', 'Head Lecturer', 'Helpdesk Consultant', 'Helper Electrical', 'Housekeeping Team Leader', 'HR Manager', 'HR Officer', 
    'HR Recruiter', 'HR Trainee', 'HVAC Consultant', 'HVAC Professional', 'Incident Controller', 'Industrial Trainee', 'Installation Technician', 
    'Instrument Technician', 'Instructor', 'Intern Trainee', 'Junior Electrical Engineer', 'Logistics Coordinator', 'Manager', 'Medical Practitioner', 
    'Orthopedic Surgeon', 'Receptionist', 'Security Officer', 'Site Engineer', 'Supervisor', 'Technician', 'Team Leader',

    # 3500 - 4000
    'IT Consultant', 'IT Executive', 'IT Secretary', 'Junior Accountant', 'Junior Web Developer', 'Junior Maintenance Engineer', 'Junior Production Executive', 
    'Junior Consultant', 'Lab Instructor', 'Lathe Machine Operator', 'Legal Intern', 'Logistics Coordinator', 'Logistics Assistant', 'Maintenance Engineer', 
    'Maintenance Manager', 'Maintenance Technician', 'Manager', 'Marketing Manager', 'NDT Technician', 'Officer', 'Orthopaedic Surgeon', 'Registrar', 
    'Service Engineer', 'Site Engineer', 'Specialist Registrar', 'Technician',

    # 4000 - 4500
    'Major Trauma Consultant', 'Course Coordinator', 'Make Up Artist', 'Makeup Artist & Hairstylist', 'Makeup Intern', 'Management Trainee', 'Manager', 
    'Safety Officer', 'Marketing Manager', 'Mechanical Foreman', 'Mechanical Technician', 'Media Group Leader', 'Medical Officer', 'Mens Groomer', 
    'MEP Designer Engineer', 'Meter Reader', 'Middle Level Executive', 'Mobile Sales Manager', 'Module Leader', 'Lecturer', 'Motivational Speaker', 
    'Trainee Engineer', 'Piping Stress Engineer', 'Junior Engineer', 'Maintenance Engineer', 'Environment Engineer', 'Warehouse Assistant', 'NDT Technician', 
    'QC Technician', 'Neurosurgeon',

    # 4500 - 5000
    'Office Assistant', 'Office Coordinator', 'Safety Officer', 'IT Consultant', 'IT Executive', 'IT Secretary', 'Onsite/Offshore Coordinator', 
    'Orthopaedic Surgeon', 'Orthopedic Consultant', 'Orthopedic Surgery Registrar', 'Personal Assistant', 'Personal Accountant', 'Piping Design Engineer', 
    'Piping Designer', 'Planning Engineer', 'Plant Manager', 'Plastic Surgeon', 'Restaurant Manager', 'Sales Retail Executive', 'Site Engineer', 'Store Officer', 
    'Supervisor', 'Technician', 'Testing Engineer', 'Medical Officer', 'Executive Engineer', 'Cashier', 'Consultant',

    # 5000 - 5500
    'Presales Engineer', 'President', 'Production Engineer', 'Production Supervisor', 'Procurement Executive', 'Process Developer', 'Programmer', 'Project Engineer', 
    'QA/QC Engineer', 'QC Inspector', 'Quality Engineer', 'Quality Executive', 'Quality Head', 'Quantity Surveyor', 'Radio Officer', 'Regional Manager', 'Registrar', 
    'Relationship Manager', 'Repair Technician', 'Report Writer', 'Resident Engineer', 'Resident Registrar', 'Residential Medical Officer', 'Restaurant Cashier', 
    'Retail Banking Officer', 'Reviewer',

    # 5500 - 6000
    'Rig Mechanic', 'Safety Engineer', 'Sales Consultant', 'Sales Representative', 'Scaffolder', 'Scaffolding Foreman', 'Scaffolding Inspector', 'Security Executive', 
    'Security Officer', 'Senior Consultant', 'Senior Design Engineer', 'Senior Medical Officer', 'Senior Piping Designer', 'Senior Safety Officer', 
    'Senior Scaffolding Supervisor', 'Senior Site Engineer', 'Service Engineer', 'Shift In-Charge', 'Site Electrical Engineer', 'Site Engineer', 'Site Inspector', 
    'Site Supervisor', 'Site Technician', 'Software Developer', 'Software Tester', 'Solar Energy Installer', 'Solar Thermal Energy Engineer', 'Specialist Registrar', 
    'Specialty Doctor', 'Spine Surgeon', 'Store Keeper', 'Supervisor', 'Technician',

    # 6000 - 6500
    'Sr. General Manager', 'Sr. HR Executive', 'Sr. Production Manager', 'Sr. Telecom Technician', 'Staff Neurosurgeon', 'Store Assistant', 'Store Incharge', 
    'Store Manager', 'Store Officer', 'Store Supervisor', 'Structure Consultant', 'Sub Investigator', 'Summer Trainee', 'Supervisor', 'Support Engineer', 
    'Support Executive', 'System Integrator', 'Tax Department Intern', 'Table Instructor', 'Tactical Security Officer', 'TBM Manager', 'Team Head', 'Team Lead', 
    'Technical Assistant', 'Technical Support Coordinator', 'Technical Voice Support', 'Technical Writer', 'Technician', 'Telecaller', 'Telecom Technician', 
    'Test Engineer', 'Testing & Commissioning Engineer', 'Testing Engineer', 'QA/QC Manager', 'Quality Control Specialist', 'Commercial Officer', 'Construction Manager', 
    'HSE Manager', 'General Manager', 'Plant Engineer', 'Lab Technician', 'Site Manager',

    # 6500 - 7000
    'Tower Officer', 'Tower Officer Supervisor', 'Trainee Engineer', 'Trainee Sonographer', 'Training Coordinator', 'Training Manager', 'Service Engineer', 
    'Estimation Engineer', 'Accounts Assistant', 'Teaching Assistant', 'VA Developer', 'Vice-President', 'Visiting Consultant', 'Visiting Faculty', 
    'Voluntary Medical Officer', 'Warehouse Assistant Supervisor', 'Welder', 'Administrative Assistant', 'Audit Assistant', 'Civil Engineer', 'Engineer', 
    'HVAC Engineer', 'Junior Engineer', 'Locum Specialist', 'Medical Officer', 'Office Assistant', 'Quality Engineer', 'Relationship Manager', 'Shift Engineer', 
    'Site Manager', 'Branch Manager', 'Mechanical Engineer', 'MEP Engineer', 'Room Technician', 'Sales Manager', 'Document Controller', 'Sales Executive', 
    'Workshop In Charge', 'Yard Manager', 'Manager',

    # PT
    "professor", "professora", "gestor", "gerente", "analista", "engenheiro",
    "engenheira", "desenvolvedor", "desenvolvedora", "consultor", "consultora",
    "coordenador", "coordenadora", "supervisor", "supervisora", "assistente",
    "técnico", "técnica", "motorista", "condutor", "nadador-salvador",
    "operador", "operadora", "piloto", "segurança",

    # ES
    "profesor", "profesora", "gerente", "analista", "ingeniero", "ingeniera",
    "desarrollador", "desarrolladora", "consultor", "consultora", "coordinador",
    "coordinadora", "supervisor", "supervisora", "asistente", "técnico",
    "técnica", "conductor", "socorrista", "operador", "operadora", "piloto",
    "seguridad",
    
    #SOFTWARE_AS_SKILL
    # Productivity / Office
    "microsoft office", "office", "microsoft", "ms office", 
    "excel", "word", "powerpoint", "outlook", "onenote",
    "google docs", "google sheets", "google slides", "google drive",
    "libreoffice", "wps office", "ms-paint","ms paint",
    "ms excel", "ms word", "ms powerpoint", "ms outlook", "ms onenote",

    # Operating systems / basic tools
    "windows", "window", "win7", "winxp", "win10", "win11",
    "linux", "ubuntu", "debian", "fedora", "centos", "red hat",
    "macos","windows 8","windows 9","windows 10","windows 11","windows 7","windows XP",
    "windows8","windows9","windows10","windows11","windows7","windowsXP",
    "window8","window9","window10","window11","window7","windowXP",
    "window 8","window 9","window 10","window 11","window 7","window XP",
    "winrar", "win-rar", "7zip", "7-zip",

    # Design / CAD / 3D
    "autocad", "auto cad","solidworks","solid works","catia", "revit", "sketchup", "fusion 360",
    "blender", "3ds max", "maya", "cinema 4d", "rhino", "rhinoceros",
    "inventor", "archicad", "microstation", "nozzle pro", "nozzlepro", 

    # Graphic design / UI / video
    "canva", "figma", "adobe xd",
    "photoshop", "adobe photoshop",
    "illustrator", "adobe illustrator",
    "indesign", "adobe indesign", "photo shop",
    "premiere pro", "adobe premiere pro",
    "after effects", "adobe after effects",
    "lightroom", "adobe lightroom",
    "davinci resolve", "final cut pro", "capcut",

    # Development environments / editors
    "vscode", "visual studio code", "visual studio",
    "pycharm", "intellij", "intellij idea",
    "eclipse", "netbeans", "codeblocks", "code::blocks",
    "android studio", "xcode", "spyder", "jupyter", "jupyter notebook",
    "notepad++", "sublime text",

    # Version control / Dev tools
    "git", "github", "gitlab", "bitbucket",
    "docker", "kubernetes", "jenkins", "ansible", "terraform",
    "postman", "swagger",

    # Databases / data tools
    "mysql", "postgresql", "postgres", "sql server", "mssql",
    "oracle", "sqlite", "mongodb", "access", "microsoft access",
    "power bi", "tableau", "qlik", "qlikview", "looker",
    "sas", "spss", "stata", "minitab", "matlab",

    # Accounting / ERP / business software
    "sap erp", "sap hana","sap",
    "oracle erp", "odoo", "primavera", "sage", "tally",
    "crm", "erp", "salesforce", "hubspot",

    # Engineering / scientific
    "ansys", "comsol", "labview", "simulink", "originlab",
    "etabs", "robot structural analysis",

    # Project / collaboration tools
    "jira", "confluence", "trello", "asana", "notion",
    "slack", "microsoft teams", "teams", "zoom",

    # Web / CMS / marketing
    "wordpress", "shopify", "woocommerce",
    "google analytics", "meta business suite",

    # GIS / mapping
    "arcgis", "qgis",

    # Cybersecurity / networking
    "wireshark", "cisco packet tracer", "packet tracer", "nmap", "xampp",


    #PROGRAMMING_LIBRARIES
    # ----------------------
    # Programming Languages
    # ----------------------
    # Core / general purpose
    "java", "python", "c", "c#", "c++", "rust", "assembly",

    # Web
    "javascript", "typescript", "html", "css", "sass", "scss",

    # Backend / enterprise
    "golang", "kotlin", "scala", "groovy",

    # .NET
    "vb.net", "f#", ".net",

    # Scripting
    "bash", "shell", "powershell", "perl",

    # Data / scientific
    "matlab", "octave", "julia",

    # Mobile
    "dart", "objective-c", "swift",

    # Functional
    "haskell", "ocaml", "erlang", "elixir", "clojure",

    # Legacy / others
    "fortran", "cobol", "pascal", "ada",

    # Game dev / niche
    "gdscript", "haxe",

    # Hardware / low-level
    "verilog", "vhdl",

    # Query / data languages
    "sql", "pl/sql", "t-sql","sql ",

    # NoSQL / query-like
    "cypher", "graphql",

    # Common misspellings / variants
    "pyhton", "javscript", "c plus plus", "c sharp",
    # ---------------------
    # Python - Data Science / ML
    # ---------------------
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "scikit-learn", "sklearn",
    "statsmodels",
    "xgboost", "lightgbm", "catboost",
    "tensorflow", "keras", "pytorch", "torch",
    "fastai","plotly", "bokeh","streamlit",

    # ---------------------
    # Python - Backend / APIs
    # ---------------------
    "django", "flask", "fastapi",
    "sqlalchemy", "alembic",
    "pydantic",

    # ---------------------
    # Python - Utils
    # ---------------------
    "beautifulsoup", "beautifulsoup4", "bs4",
    "pytest","celery", "beautiful soup",

    # ---------------------
    # JavaScript / Frontend
    # ---------------------
    "react.js", "reactjs",
    "vue", "vue.js", "vuejs",
    "angular", "angularjs",
    "svelte",
    "next.js", "nextjs",
    "nuxt.js", "nuxtjs",

    # UI frameworks
    "bootstrap", "tailwind", "tailwindcss",
    "material-ui",
    "chakra-ui",

    # ---------------------
    # JavaScript / Backend
    # ---------------------
    "node.js", "nodejs",
    "express.js",
    "nestjs",
    "socket.io",

    # ---------------------
    # Java / JVM
    # ---------------------
    "spring boot", "springboot",
    "hibernate",

    # ---------------------
    # .NET
    # ---------------------
    ".net", "asp.net", "aspnet",
    "entity framework", "ef core",

    # ---------------------
    # PHP
    # ---------------------
    "laravel", "symfony", "codeigniter",

    # ---------------------
    # Ruby
    # ---------------------
    "ruby on rails", "rails",

    # ---------------------
    # Databases / ORMs / connectors
    # ---------------------
    "mysql connector", "mysql-connector", "mysqlextra",
    "psycopg2",
    "sqlite3",
    "mongodb driver", "mongoose",
    "redis",

    # ---------------------
    # Data engineering
    # ---------------------
    "pyspark", 
    "airflow",

    # ---------------------
    # Testing
    # ---------------------
    "jest", "mocha", "chai",
    "junit",
    "selenium",

    # ---------------------
    # Misc / common libs
    # ---------------------
    "lodash",
    "axios",
    "moment.js", "moment",
    "three.js", "d3.js", "chart.js",
    

    #Mais
    # job titles / roles
    "civil engineer",
    "civil site engineer",
    "electrical technician",
    "onshore electrical technician",
    "assistant professor",
    "intern developer",
    "junior developer",
    "draftsman",
    "cad draftsman",
    "market research specialist",
    "market analyst",
    "quality in-charge",
    "facility engineer",
    "piping stress engineer",
    "multi technician",
    "safety officer",
    "site engineer",
    "technician",
    "drafting manager",

    # extra software / technical tools
    "arduino",
    "navis works",
    "navisworks",
    "psim",
    "pv-elite",
    "pv elite",
    "caesar ii",
    "caesar-ii",
    "caesar ii piping stress analysis",
    "tally 9.0",
    "tally erp 9",
    "ms-office",
    "ms office",

    # engineering / technical concepts
    "deep learning",
    "machine learning",
    "fault tree analysis",
    "harmonic analysis",
    "short circuit",
    "voltage drop calculation",
    "microprocessor based relay",
    "numerical relay",
    "circuit breaker",
    "circuit breaker panel board",
    "welding",
    "dc welding",
    "brazing",
    "fabricator",
    "method statement",
    "preliminary stress analysis",
    "soil testing",
    "initial final testing",
    "concrete consistence test",
    "road levelling",
    "pipe line",
    "pipeline",
    "reinforcement",
    "foundation",
    "shuttering",
    "floor finishing",
    "finishing work",
    "plate rolling",

    # specific useful skills / methodologies
    "8d approach",
    "6sigma",
    "5s",
    "5s activities",
    "cpk",
    "poka yoke technique",

    ]

# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

result = compare_dataset_entities_vs_gliner(
    json_folder=r"C:\Users\danie\Downloads\ResumesJsonAnnotated\ResumesJsonAnnotated",
    start=1,
    end=1000,
    dataset_output_file=EVALUATION_TXT_DIR / "dataset_entities_filtered_unique_freq.txt",
    gliner_output_file=EVALUATION_TXT_DIR / "gliner_entities_plus_regex_languages_unique_freq.txt",
    regex_output_file=EVALUATION_TXT_DIR / "regex_languages_only_unique_freq.txt",
    similar_output_file=EVALUATION_TXT_DIR / "dataset_vs_gliner_matched_entities.txt",
    different_output_file=EVALUATION_TXT_DIR / "dataset_vs_gliner_unmatched_groups.txt",
    dataset_only_output_file=EVALUATION_TXT_DIR / "dataset_only_unmatched_entities.txt",
    gliner_only_output_file=EVALUATION_TXT_DIR / "gliner_only_unmatched_entities.txt",
    summary_output_file=EVALUATION_TXT_DIR / "summary_report.txt",
    exact_stopwords=stopwords_exatas,
    partial_stopwords=stopwords_parciais,
    trusted_text_entities_as_true_positive= trusted_words,
    gliner_threshold=0.15,
    gliner_skills_column="skills",
    gliner_job_titles_column="job_title",
    sort_by="frequency",
    enable_regex_language_extraction=True,
    trust_regex_languages_as_true_positive=True,
    min_frequency_filters={"dataset": 0, "gliner": 0, "regex": 0},
)