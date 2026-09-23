import re
import os
import unicodedata
import difflib
from app.utils.logger import logger
from app.storage.lancedb_store import get_hash_table
from app.query.query_plan import (
    QueryPlan,
    QueryIntent,
    QueryOperation,
    FaceIntent,
    RequestScope,
    Modality,
    SourceSpec,
    QueryFilters,
)
from app.search.face_search import (
    is_face_search_query,
    extract_person_name_from_question,
)

# ============================================================
# QUERY INTENT TAXONOMY CONSTANTS
# ============================================================
INTENT_SOURCE_LOOKUP = "SOURCE_LOOKUP"
INTENT_SOURCE_TYPE = "SOURCE_TYPE"
INTENT_FILE_METADATA = "FILE_METADATA"
INTENT_FILE_COUNT = "FILE_COUNT"
INTENT_SPREADSHEET_QUERY = "SPREADSHEET_QUERY"
INTENT_FILE_LIST = "FILE_LIST"
INTENT_TEMPORAL_FILE_QUERY = "TEMPORAL_FILE_QUERY"
INTENT_TEXT_SEARCH = "TEXT_SEARCH"
INTENT_IMAGE_FILENAME_QUERY = "IMAGE_FILENAME_QUERY"
INTENT_IMAGE_COUNT_QUERY = "IMAGE_COUNT_QUERY"
INTENT_IMAGE_OCR = "IMAGE_OCR"
INTENT_IMAGE_TEXT_EXTRACTION = "IMAGE_TEXT_EXTRACTION"
INTENT_IMAGE_SUMMARY = "IMAGE_SUMMARY"
INTENT_IMAGE_VISUAL_QUERY = "IMAGE_VISUAL_QUERY"
INTENT_IMAGE_FACE_QUERY = "IMAGE_FACE_QUERY"
INTENT_DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
INTENT_DOCUMENT_TEXT_QUERY = "DOCUMENT_TEXT_QUERY"
INTENT_CORRECTION = "CORRECTION"

GENERIC_MEDIA_TERMS = {
    "image", "images", "picture", "pictures", "photo", "photos",
    "file", "files", "document", "documents", "pdf", "docx", "xlsx",
    "csv", "txt", "excel", "spreadsheet", "spreadsheets",
    "list", "name", "names", "page", "pages",
    "content", "contents", "summary", "summaries",
    "note", "notes", "doc", "docs", "paper", "papers",
    "sheet", "sheets", "text", "texts", "log", "logs",
    "data", "engineering", "company", "companies", "revenue",
    "diagram", "diagrams", "chart", "charts", "architecture",
    "written", "report", "project", "analysis", "system", "code"
}

STOP_WORDS_SET = {
    "the", "a", "an", "of", "in", "from", "to", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "for", "with",
    "on", "at", "by", "this", "that", "my", "please", "what", "who", "how", "which",
    "where", "when", "summarize", "summarise", "summary", "chapter", "section", "part",
    "show", "get", "give", "said", "discussed", "listen", "recording", "recordings", "content", "me",
    "can", "you", "listed", "mention", "mentioned", "many", "much", "count", "number",
    "there", "tell", "read", "named", "name", "names", "called", "titled", "images", "image", "file", "files",
    "all", "entire", "complete", "everything", "whole", "full", "spoken", "speech", "talk", "talking",
    "translate", "convert", "translation", "conversion", "folder", "directory", "drive", "recent", "latest", "newest",
    "into", "onto", "within", "without", "between", "among", "under", "over", "through", "during", "before", "after", "about",
    "line", "lines", "phrase", "phrases", "word", "words", "sentence", "sentences", "item", "items",
    "english", "hindi", "hinglish", "french", "german", "spanish", "italian", "portuguese", "japanese", "chinese", "russian",
    "want", "wants", "wanted", "like", "likes", "liked", "would", "should", "could", "need", "needs", "needed",
    "require", "requires", "required", "try", "trying", "ask", "asking", "asked", "know", "find", "search", "lookup",     "main", "points", "point", "discuss", "discussion",
    "discussions", "happened", "happen", "talked",
    "talk", "conversation", "conversations", "topic", "topics"
}


def get_indexed_filenames():
    """
    Helper function to retrieve all indexed filenames from the processed_files metadata table.
    Returns a list of filename strings.
    """
    hash_table = get_hash_table()
    if hash_table is None:
        return []

    database_dataframe = hash_table.to_pandas()
    if database_dataframe.empty:
        return []

    filenames_list = []
    if "path" in database_dataframe.columns:
        for path_string in database_dataframe["path"].dropna().tolist():
            base_filename = os.path.basename(path_string)
            if base_filename and base_filename not in filenames_list:
                filenames_list.append(base_filename)

    return filenames_list


def split_camel_case(text_input):
    """
    Splits CamelCase, PascalCase, and Acronym-word transitions in filenames and queries.
    Example: 'SQLNotesForProfessionals' -> 'SQL Notes For Professionals'
             'PythonCheatSheet' -> 'Python Cheat Sheet'
    """
    if not text_input:
        return ""
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text_input))
    s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s1)
    return s2


def normalize_string(text_input):
    """
    Normalizes text by splitting CamelCase, lowercasing, NFKD unicode normalization,
    stripping apostrophes/possessives, and replacing punctuation/symbols with single spaces.
    Example: "SQLNotesForProfessionals.pdf" -> "sql notes for professionals pdf"
             "Behari_lal-call.m4a" -> "behari lal call m4a"
    """
    if not text_input:
        return ""
    
    # 0. Split CamelCase / PascalCase
    camel_split = split_camel_case(text_input)

    # 1. Unicode normalization (NFKD)
    normalized_unicode = unicodedata.normalize("NFKD", str(camel_split))
    
    # 2. Lowercase
    lowercased_text = normalized_unicode.lower()
    
    # 3. Strip possessive apostrophes ('s, ’s)
    text_without_possessive = re.sub(r"['’]s\b", "", lowercased_text)
    
    # 4. Replace punctuation with single spaces
    cleaned_text = re.sub(r"[_\-\.\:\,\!\?\/'\"`]+", " ", text_without_possessive)
    
    # 5. Normalize whitespace
    return " ".join(cleaned_text.split())


def compact_alphanumeric(text_input):
    """
    Strips all non-alphanumeric characters to allow robust matching across digit variants.
    Example: 'dense 2', 'dense_2', 'dense-2' -> 'dense2'
    """
    if not text_input:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text_input).lower())


def extract_entity_tokens(text_input):
    """
    Extracts non-stopword entity tokens from a text input using normalized text.
    """
    normalized = normalize_string(text_input)
    raw_tokens = normalized.split()
    filtered_tokens = []
    for token in raw_tokens:
        if len(token) >= 2 and token not in STOP_WORDS_SET and token not in GENERIC_MEDIA_TERMS:
            filtered_tokens.append(token)

    return filtered_tokens


def calculate_token_similarity(token_a, token_b):
    """
    Calculates sequence similarity ratio between two tokens.
    """
    if not token_a or not token_b:
        return 0.0
    if token_a == token_b:
        return 1.0
    return difflib.SequenceMatcher(None, token_a, token_b).ratio()


def find_best_matching_source(question_lower, indexed_files, query_modality="all", face_intent="none"):
    """
    Data-driven typo-tolerant filename and source resolution pipeline.
    Preserves exact original filename casing from indexed_files.
    
    Resolution Strategy:
    1. Exact full filename in query (e.g. Accounts.m4a, audio.mpeg).
    2. Entity token matching (exact, normalized, and typo-tolerant fuzzy matching).
    3. Exact normalized stem match (handling digit variants like 'dense 2' -> 'dense2.webp').
    4. Compact stem substring fallback.
    
    Returns tuple: (resolved_candidate_filename, confidence_score)
    """
    if not indexed_files:
        return None, 0.0

    image_extensions = (".jpg", ".jpeg", ".png", ".webp")

    # Filter candidate files by modality if specified
    candidate_files = []
    for indexed_file in indexed_files:
        filename_lower = indexed_file.lower()
        if query_modality == "image" or face_intent in ["face_search", "face_identification"]:
            if filename_lower.endswith(image_extensions):
                candidate_files.append(indexed_file)
        else:
            candidate_files.append(indexed_file)

    if not candidate_files:
        return None, 0.0
    normalized_question = normalize_string(question_lower)
    compact_question = compact_alphanumeric(question_lower)

    # Stage 1: Exact Full Filename Matching
    # Compare the COMPLETE normalized filename against the COMPLETE
    # normalized query so multi-word filenames are never truncated.
    for candidate in candidate_files:
        candidate_normalized = normalize_string(candidate)
        if candidate_normalized and candidate_normalized in normalized_question:
            return candidate, 1.0

    # Also support the literal filename exactly as stored.
    for candidate in candidate_files:
        if candidate.lower() in question_lower:
            return candidate, 1.0

    # Stage 1.5: Natural Stem Sub-phrase Matching
    # Check if a multi-word or single-word entity phrase from the query matches candidate filename stems as a sub-phrase
    # Example: "SQL notes" matches stem "SQLNotesForProfessionals" -> "sql notes for professionals"
    natural_stem_matches = []
    for candidate in candidate_files:
        file_stem = os.path.splitext(candidate)[0]
        normalized_stem = normalize_string(file_stem)

        # Split stem words, ignoring generic prepositions
        stem_words = [w for w in normalized_stem.split() if w not in {"a", "an", "the", "of", "in", "for", "to", "and", "or", "is", "by", "with", "on", "at"}]

        if len(stem_words) >= 1:
            for sub_len in range(len(stem_words), 0, -1):
                for start_idx in range(len(stem_words) - sub_len + 1):
                    sub_phrase = " ".join(stem_words[start_idx : start_idx + sub_len])
                    if len(sub_phrase) >= 3 and sub_phrase not in GENERIC_MEDIA_TERMS:
                        boundary_pattern = r"\b" + re.escape(sub_phrase) + r"\b"
                        if re.search(boundary_pattern, normalized_question):
                            natural_stem_matches.append((sub_len, len(sub_phrase), candidate))
                            break

    if natural_stem_matches:
        natural_stem_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_sub_len = natural_stem_matches[0][0]
        top_sub_char_len = natural_stem_matches[0][1]

        top_candidates = list(set([m[2] for m in natural_stem_matches if m[0] == top_sub_len and m[1] == top_sub_char_len]))

        if len(top_candidates) == 1:
            return top_candidates[0], 0.95
        elif len(top_candidates) > 1:
            return top_candidates, 0.60

    # Stage 2: Entity Token Matching (Exact + Fuzzy Match)
    question_entity_tokens = extract_entity_tokens(question_lower)

    if question_entity_tokens:
        scored_candidates = []
        for candidate in candidate_files:
            file_stem = os.path.splitext(candidate)[0]
            stem_tokens = extract_entity_tokens(file_stem)

            best_token_score = 0.0
            exact_match_count = 0

            generic_topic_words = GENERIC_MEDIA_TERMS

            for q_token in question_entity_tokens:
                if q_token in generic_topic_words:
                    continue

                for s_token in stem_tokens:
                    if s_token in generic_topic_words:
                        continue

                    if q_token == s_token:
                        exact_match_count += 1
                        best_token_score = max(best_token_score, 1.0)
                        break

                    # Fuzzy match check for genuine typos in longer names (e.g. jethlal vs jethalal)
                    similarity = calculate_token_similarity(q_token, s_token)
                    if len(q_token) >= 5 and len(s_token) >= 5:
                        if similarity >= 0.82:
                            best_token_score = max(best_token_score, similarity)

            # A single matched token only resolves if it matches a high fraction of the stem or is exact
            if exact_match_count > 0:
                score = 0.90 if exact_match_count >= 2 else (0.85 if len(stem_tokens) <= 2 else 0.50)
                scored_candidates.append((score, exact_match_count, candidate))
            elif best_token_score >= 0.82:
                scored_candidates.append((best_token_score, 0, candidate))

        if scored_candidates:
            scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            top_score = scored_candidates[0][0]
            if top_score >= 0.70:
                return scored_candidates[0][2], top_score

    # Stage 3: Exact Stem & Compact Stem Matching
    best_exact_match = None
    best_exact_length = 0

    for candidate in candidate_files:
        file_stem = os.path.splitext(candidate)[0]
        normalized_stem = normalize_string(file_stem)
        compact_stem = compact_alphanumeric(file_stem)

        if normalized_stem in GENERIC_MEDIA_TERMS:
            continue

        word_boundary_pattern = r"\b" + re.escape(normalized_stem) + r"\b"
        if re.search(word_boundary_pattern, normalized_question):
            if len(normalized_stem) > best_exact_length:
                best_exact_match = candidate
                best_exact_length = len(normalized_stem)

        if not best_exact_match and compact_stem and len(compact_stem) >= 3:
            if compact_stem == compact_question or re.search(r"\b" + re.escape(normalized_stem) + r"\b", normalized_question):
                return candidate, 0.9

            digit_match = re.search(r"(\d+)$", file_stem)
            if digit_match:
                digit_number = digit_match.group(1)
                base_part = file_stem[:len(file_stem) - len(digit_number)]
                norm_base = normalize_string(base_part)
                digit_pattern = r"\b" + re.escape(norm_base) + r"\s*" + re.escape(digit_number) + r"\b"
                if re.search(digit_pattern, normalized_question):
                    return candidate, 0.95

    if best_exact_match:
        return best_exact_match, 0.95

    # Stage 4: Substring Fallback for compact stems
    for candidate in candidate_files:
        file_stem = os.path.splitext(candidate)[0]
        compact_stem = compact_alphanumeric(file_stem)
        if compact_stem and len(compact_stem) >= 4 and compact_stem not in GENERIC_MEDIA_TERMS and compact_stem in compact_question:
            return candidate, 0.8

    return None, 0.0


def resolve_semantic_source(question_lower, indexed_files, query_modality="all"):
    """
    Performs Semantic Source Resolution BEFORE content retrieval and intent routing.
    Queries indexed chunks in documents and image_documents tables using BM25 and CrossEncoder reranking,
    aggregates chunk similarity scores per candidate source file, and evaluates confidence/margin thresholds.

    Returns tuple: (resolved_source, top_score, second_score, candidate_list)
      - resolved_source: str or None (basename of resolved source file)
      - top_score: float (top file aggregated score)
      - second_score: float (runner-up file aggregated score)
      - candidate_list: list of str (candidate basenames when ambiguous)
    """
    if not indexed_files:
        return None, 0.0, 0.0, []

    try:
        from app.storage.lancedb_store import get_table, get_image_table
        from app.search.vector_search import is_noise_chunk, rerank, get_words
        from rank_bm25 import BM25Okapi
        import math

        doc_table = get_table()
        image_table = get_image_table()

        all_indexed_rows = []

        if doc_table is not None and doc_table.count_rows() > 0:
            df_doc = doc_table.to_pandas()
            for i in range(len(df_doc)):
                r = df_doc.iloc[i]
                all_indexed_rows.append({
                    "chunk_id": str(r.get("chunk_id", f"doc_{i}")),
                    "path": str(r.get("path", "")),
                    "file_type": str(r.get("file_type", "document")),
                    "text": str(r.get("text", ""))
                })

        if image_table is not None and image_table.count_rows() > 0:
            df_img = image_table.to_pandas()
            for i in range(len(df_img)):
                r = df_img.iloc[i]
                all_indexed_rows.append({
                    "chunk_id": str(r.get("chunk_id", f"img_{i}")),
                    "path": str(r.get("path", "")),
                    "file_type": "image",
                    "text": str(r.get("text", ""))
                })

        if not all_indexed_rows:
            return None, 0.0, 0.0, []

        image_extensions = (".jpg", ".jpeg", ".png", ".webp")

        filtered_rows = []
        for row in all_indexed_rows:
            r_base = os.path.basename(row["path"]).lower()
            if is_noise_chunk(row["text"]):
                continue
            if query_modality == "image":
                if r_base.endswith(image_extensions):
                    filtered_rows.append(row)
            else:
                filtered_rows.append(row)

        if not filtered_rows:
            return None, 0.0, 0.0, []

        tokenized_corpus = [get_words(r["text"]) for r in filtered_rows]
        if not any(tokenized_corpus):
            return None, 0.0, 0.0, []

        query_tokens = get_words(question_lower)
        if not query_tokens:
            return None, 0.0, 0.0, []

        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]
        candidate_chunks = [filtered_rows[i] for i in top_indices if bm25_scores[i] > 0]

        if not candidate_chunks:
            return None, 0.0, 0.0, []

        scored_chunks = rerank(question_lower, candidate_chunks)

        file_scores = {}
        for chunk, score in scored_chunks:
            file_base = os.path.basename(chunk["path"])
            norm_score = 1.0 / (1.0 + math.exp(-score))
            if file_base not in file_scores:
                file_scores[file_base] = []
            file_scores[file_base].append(norm_score)

        if not file_scores:
            return None, 0.0, 0.0, []

        aggregated_scores = []
        for f_base, scores in file_scores.items():
            top_c_score = max(scores)
            avg_c_score = sum(scores) / len(scores)
            chunk_score = 0.7 * top_c_score + 0.3 * avg_c_score

            # Filename stem token overlap score
            file_stem = os.path.splitext(f_base)[0]
            stem_tokens = extract_entity_tokens(file_stem)
            q_tokens = extract_entity_tokens(question_lower)

            name_overlap_score = 0.0
            if q_tokens and stem_tokens:
                overlap_count = sum(1 for qt in q_tokens if any(qt == st or (len(qt) >= 2 and qt in st) for st in stem_tokens))
                name_overlap_score = overlap_count / len(q_tokens)

            # Check if query contains explicit entity noun tokens absent from candidate stem (e.g. 'physics' in query vs 'SQLNotes' candidate)
            has_conflicting_entity = False
            if q_tokens and stem_tokens:
                for qt in q_tokens:
                    if len(qt) >= 4 and qt not in GENERIC_MEDIA_TERMS:
                        if not any(qt in st or st in qt for st in stem_tokens):
                            has_conflicting_entity = True
                            break

            final_file_score = 0.50 * chunk_score + 0.35 * name_overlap_score + 0.15 * (1.0 if chunk_score > 0 else 0.0)
            if has_conflicting_entity:
                final_file_score *= 0.3

            aggregated_scores.append((f_base, final_file_score))

        aggregated_scores.sort(key=lambda item: item[1], reverse=True)

        top_file, top_score = aggregated_scores[0]
        second_score = aggregated_scores[1][1] if len(aggregated_scores) > 1 else 0.0
        candidates = [f[0] for f in aggregated_scores if f[1] >= 0.45]

        return top_file, top_score, second_score, candidates
    except Exception as err:
        logger.exception("Semantic source resolution failed: %s", err)
        return None, 0.0, 0.0, []


def resolve_canonical_source_id(source_hint):
    """
    Resolves a source_hint filename string to the canonical full filepath in LanceDB index or filesystem.
    """
    if not source_hint or str(source_hint).startswith("UNRESOLVED_"):
        return None

    hash_table = get_hash_table()
    if hash_table is not None and hash_table.count_rows() > 0:
        database_dataframe = hash_table.to_pandas()
        if not database_dataframe.empty and "path" in database_dataframe.columns:
            for path_string in database_dataframe["path"].dropna().tolist():
                if os.path.basename(path_string).lower() == source_hint.lower():
                    return path_string
                if source_hint.lower() in path_string.lower():
                    return path_string

    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    watched_directory = os.path.join(base_directory, "watched_folder")
    if os.path.exists(watched_directory):
        for filename in os.listdir(watched_directory):
            if filename.lower() == source_hint.lower():
                return os.path.join(watched_directory, filename)

    for filename in os.listdir(base_directory):
        if filename.lower() == source_hint.lower():
            return os.path.join(base_directory, filename)

    return None


def extract_target_language(text_input):
    """
    Generic language target extraction function.
    Parses queries for explicit language conversion targets (e.g. 'in english', 'to french', 'in spanish').
    Returns language name string or None.
    """
    if not text_input:
        return None

    lowercased_text = text_input.lower()
    supported_languages = [
        "english", "hindi", "hinglish", "french", "german",
        "spanish", "italian", "portuguese", "japanese", "chinese", "russian"
    ]

    for language_name in supported_languages:
        # Check explicit preposition patterns like "in english", "to english", "into english"
        in_pattern = r"\bin\s+" + re.escape(language_name) + r"\b"
        to_pattern = r"\bto\s+" + re.escape(language_name) + r"\b"
        into_pattern = r"\binto\s+" + re.escape(language_name) + r"\b"

        if re.search(in_pattern, lowercased_text) or re.search(to_pattern, lowercased_text) or re.search(into_pattern, lowercased_text):
            return language_name

    return None


def infer_query_operation(
    question: str,
    intent: str,
    request_scope: str,
) -> QueryOperation:
    """
    Determine what the user wants done with the resolved source/data.

    Intent answers WHAT KIND OF QUERY this is.
    Operation answers WHAT ACTION should be performed.
    """

    q = question.lower().strip()

    if intent == INTENT_CORRECTION:
        return QueryOperation.ANSWER

    if intent in {
        INTENT_FILE_COUNT,
        INTENT_IMAGE_COUNT_QUERY,
    }:
        return QueryOperation.COUNT

    if intent in {
        INTENT_FILE_LIST,
        INTENT_TEMPORAL_FILE_QUERY,
        INTENT_SOURCE_LOOKUP,
    }:
        return QueryOperation.LIST

    if intent == "FULL_CONTENT_FETCH":
        return QueryOperation.FETCH

    if intent in {
        INTENT_DOCUMENT_SUMMARY,
        INTENT_IMAGE_SUMMARY,
    }:
        return QueryOperation.SUMMARIZE

    if intent == INTENT_IMAGE_FILENAME_QUERY:
        return QueryOperation.DISPLAY

    if intent == INTENT_SPREADSHEET_QUERY:

        if re.search(
            r"\b(?:sort|sorted|order|ordered|rank|highest|lowest|"
            r"largest|smallest|maximum|minimum|top|bottom)\b",
            q,
        ):
            return QueryOperation.SORT

        if re.search(
            r"\b(?:group|grouped|per|each|every)\b",
            q,
        ):
            return QueryOperation.GROUP

        if re.search(
            r"\b(?:how\s+many|number\s+of|count)\b",
            q,
        ):
            return QueryOperation.COUNT

        if re.search(
            r"\b(?:list|show|display|give|provide)\b",
            q,
        ):
            if re.search(
                r"\b(?:in|from|with|having|for|where|"
                r"backend|frontend|full[\s-]?stack|product|"
                r"analytics|gcc|ctc|salary|location|role|type)\b",
                q,
            ):
                return QueryOperation.FILTER

            return QueryOperation.LIST

        return QueryOperation.FILTER

    if intent in {
        INTENT_IMAGE_VISUAL_QUERY,
        INTENT_IMAGE_OCR,
        INTENT_IMAGE_TEXT_EXTRACTION,
        INTENT_IMAGE_FACE_QUERY,
    }:
        return QueryOperation.ANSWER

    if intent == INTENT_SOURCE_TYPE:
        return QueryOperation.ANSWER

    return QueryOperation.ANSWER


def analyze_query(question, indexed_files=None, active_file=None, request_modality=None):
    """
    Analyzes the user query and produces structured classification output containing:
    - intent: explicit Intent Taxonomy constant
    - temporal_intent: temporal category
    - face_intent: face processing category
    - modality: audio | image | video | pdf | docx | spreadsheet | text | all
    - source_hint: resolved filename string if query specifies a source
    - canonical_source_id: full resolved path to source file
    - unresolved_explicit_source: True if query requested explicit source which failed resolution
    - target_language: explicit language conversion target if specified in query
    """
    question_lower = question.strip().lower()

    if indexed_files is None:
        indexed_files = get_indexed_filenames()

    # Step 1: Check Correction Intent
    correction_patterns = [
        r"\byou\s+are\s+incorrect\b", r"\byou\s+are\s+wrong\b", r"\bwrong\s+answer\b",
        r"\bwrong\s+file\b", r"\bwrong\s+source\b", r"\bthis\s+is\s+incorrect\b",
        r"\bthat\s+is\s+not\s+correct\b", r"\bcorrect\s+it\b", r"\bplease\s+correct\b",
        r"\bthe\s+correct\s+file\s+is\b", r"\bthe\s+source\s+should\s+be\b",
        r"\bnot\s+this\s+file\b", r"\bnot\s+this\b", r"\buse\s+.*\s+instead\b"
    ]
    is_correction = any(re.search(pat, question_lower) for pat in correction_patterns)
    if not is_correction:
        from app.services.interaction_state import get_last_interaction
        last_state = get_last_interaction()
        if last_state and last_state.get("previous_query"):
            if any(phrase in question_lower for phrase in ["but there are", "there are two", "one is", "actually", "wrong"]):
                is_correction = True

    # Step 2: Generic Metadata / File Inventory & Temporal Intent Extraction
    from app.utils.date_parser import parse_date_expression
    start_dt, end_dt, date_label = parse_date_expression(question_lower)

    file_list_patterns = [
        r"\blist\s+(?:all\s+)?files?\b", r"\bshow\s+(?:all\s+)?files?\b", r"\blist\s+(?:all\s+)?images?\b", r"\bshow\s+(?:all\s+)?images?\b",
        r"\blist\s+(?:all\s+)?photos?\b", r"\bshow\s+(?:all\s+)?photos?\b", r"\blist\s+(?:all\s+)?pictures?\b", r"\bshow\s+(?:all\s+)?pictures?\b",
        r"\blist\s+(?:all\s+)?audio\b", r"\bshow\s+(?:all\s+)?audio\b", r"\blist\s+(?:all\s+)?documents?\b", r"\bshow\s+(?:all\s+)?documents?\b",
        r"\blist\s+(?:all\s+)?pdfs?\b", r"\bshow\s+(?:all\s+)?pdfs?\b",
        r"\blist\s+(?:all\s+)?(?:spreadsheets?|workbooks?|excel(?:\s+files?)?|xlsx|xls|csv)\b",
        r"\bshow\s+(?:all\s+)?(?:spreadsheets?|workbooks?|excel(?:\s+files?)?|xlsx|xls|csv)\b",
        r"\bwhich\s+files\b", r"\bwhich\s+audio\b", r"\bwhich\s+images?\b", r"\bwhich\s+photos?\b", r"\bwhich\b.*\b(?:files?|images?|photos?|pictures?|audio|recordings?|documents?)\b",
        r"\bfiles?\s+uploaded\b", r"\bfiles?\s+added\b", r"\bimages?\s+added\b", r"\bphotos?\s+added\b", r"\baudio\s+added\b", r"\brecordings?\s+added\b",
        r"\brecent\s+files?\b", r"\blatest\s+files?\b", r"\bnewest\s+files?\b", r"\bmost\s+recent\b", r"\blast\s+files?\b",
        r"\brecently\s+added\b", r"\bnewly\s+added\b", r"\bdate-wise\b", r"\bdatewise\b", r"\bby\s+date\b",
        r"\b(?:top|latest|recent|first|last)\s+\d+\s*(?:files?|images?|photos?|pictures?|audio|recordings?|documents?)\b",
        r"\b\d+\s+(?:recent|latest|last)\s+(?:files?|images?|photos?|pictures?|audio|recordings?|documents?)\b",
        r"\bwhat\s+(?:audio\s+|image\s+|photo\s+)?files\b", r"\bwhat\s+files\s+(?:are\s+in|do\s+i\s+have)\b",
        r"\bnames?\s+of\s+(?:my\s+)?(?:audio\s+|image\s+|photo\s+)?files?\b",
        r"\brecordings?\s+in\s+(?:my\s+)?folder\b", r"\bfiles?\s+in\s+(?:my\s+)?folder\b",
        r"\bshow\s+(?:me\s+)?(?:all\s+)?audio\s+recordings?\b", r"\bshow\s+(?:every\s+)?sound\s+file\b", r"\ball\s+audio\s+recordings?\b"
    ]
    is_file_list_query = any(re.search(pat, question_lower) for pat in file_list_patterns) or (start_dt is not None)

    count_temporal_patterns = [
        r"\bhow\s+many\s+files\b", r"\bhow\s+many\s+images\b", r"\bhow\s+many\s+photos\b",
        r"\bhow\s+many\s+pictures\b", r"\bhow\s+many\s+audio\b", r"\bhow\s+many\s+recordings\b",
        r"\bhow\s+many\s+documents\b", r"\bcount\s+of\s+files\b", r"\bcount\s+of\s+images\b",
        r"\bcount\s+of\s+photos\b", r"\bcount\s+of\s+audio\b"
    ]
    is_count_temporal_query = any(re.search(pat, question_lower) for pat in count_temporal_patterns)

    # Dynamic limit extraction (e.g. 'latest 5 files', '5 recent images', 'top 10 files')
    question_without_date_words = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}\b",
        "",
        question_lower
    )
    question_without_date_words = re.sub(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        "",
        question_without_date_words
    )
    question_without_date_words = re.sub(r"\b\d{4}\b", "", question_without_date_words)

    explicit_limit_match = re.search(
        r"\b(?:top|latest|recent|first|last)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:recent|latest|last|files?|images?|audio|documents?)\b",
        question_without_date_words
    )
    extracted_limit = None
    if explicit_limit_match:
        number_string = explicit_limit_match.group(1) or explicit_limit_match.group(2)
        if number_string:
            extracted_limit = int(number_string)

    # Default limit to 5 if user asked for recent/latest files without specifying an explicit limit
    if extracted_limit is None and any(word in question_lower for word in ["recent", "latest", "newest", "last"]):
        extracted_limit = 5

    temporal_intent = "none"
    if is_count_temporal_query:
        temporal_intent = INTENT_FILE_COUNT
    elif is_file_list_query:
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY

    # Step 3: Modality & File Type Categorization
    modality = "all"
    if request_modality and request_modality.lower() != "all":
        modality = request_modality.lower().strip()
    elif active_file:
        active_ext = os.path.splitext(active_file)[1].lower()
        if active_ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}:
            modality = "image"
        elif active_ext in {".xls", ".xlsx", ".csv", ".ods"}:
            modality = "spreadsheet"
        elif active_ext in {".pdf"}:
            modality = "pdf"
        elif active_ext in {".doc", ".docx"}:
            modality = "docx"

    image_keywords = [
        "image", "images",
        "picture", "pictures",
        "photo", "photos",
        "visual",
        "diagram",
        "chart",
        "figure",
        "jpg", "jpeg", "png", "webp"
    ]

    pdf_keywords = ["pdf", "pdfs"]

    spreadsheet_keywords = [
        "excel", "spreadsheet", "spreadsheets", "workbook", "workbooks",
        "xlsx", "xls", "csv", "ods", "sheet", "sheets"
    ]

    doc_keywords = [
        "docx", "doc", "document", "documents",
        "paper", "papers", "notes", "note"
    ]

    if modality == "all":
        if any(kw in question_lower for kw in image_keywords):
            modality = "image"
        elif any(kw in question_lower for kw in pdf_keywords):
            modality = "pdf"
        elif any(kw in question_lower for kw in spreadsheet_keywords):
            modality = "spreadsheet"
        elif any(kw in question_lower for kw in doc_keywords):
            modality = "docx"

    # ============================================================
    spreadsheet_context_patterns = [
        r"\bexcel\b",
        r"\bspreadsheet\b",
        r"\bspreadsheets\b",
        r"\bworkbook\b",
        r"\bworkbooks\b",
        r"\bsheet\b",
        r"\bsheets\b",
        r"\bxlsx\b",
        r"\bxls\b",
        r"\bcsv\b",
    ]

    spreadsheet_operation_patterns = [
        r"\blist\b",
        r"\bshow\b",
        r"\bprovide\b",
        r"\bgive\b",
        r"\bdisplay\b",
        r"\bfind\b",
        r"\bfilter\b",
        r"\bsort\b",
        r"\border\b",
        r"\brank\b",
        r"\bgroup\b",
        r"\bgrouped\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\ball\b",
        r"\bhow\s+many\b",
        r"\bnumber\s+of\b",
        r"\bcount\b",
        r"\bin\s+each\b",
        r"\bfor\s+each\b",
        r"\bper\b",
        r"\bby\b",
    ]

    spreadsheet_field_patterns = [
        r"\bcompany\b",
        r"\bcompanies\b",
        r"\bbackend\b",
        r"\bfront[\s-]?end\b",
        r"\bfull[\s-]?stack\b",
        r"\bproduct\b",
        r"\banalytics\b",
        r"\bgcc\b",
        r"\bctc\b",
        r"\bpackage\b",
        r"\blpa\b",
        r"\bsalary\b",
        r"\brole\b",
        r"\broles\b",
        r"\blocation\b",
        r"\barea\b",
        r"\bcity\b",
        r"\bcities\b",
        r"\btype\b",
    ]

    has_spreadsheet_context = any(
        re.search(pattern, question_lower, re.IGNORECASE)
        for pattern in spreadsheet_context_patterns
    )

    has_spreadsheet_operation = any(
        re.search(pattern, question_lower, re.IGNORECASE)
        for pattern in spreadsheet_operation_patterns
    )

    has_spreadsheet_field = any(
        re.search(pattern, question_lower, re.IGNORECASE)
        for pattern in spreadsheet_field_patterns
    )

    # A query is treated as a structured spreadsheet query when:
    #
    #   1. it explicitly mentions spreadsheet/table terminology and
    #      contains an operation/field, OR
    #
    #   2. it is clearly a structured company-table operation using
    #      spreadsheet fields such as CTC, package, backend, product,
    #      city, location, etc.
    #
    # The second condition is important because users should not have
    # to say "Excel" in every query when the active source is an Excel
    # workbook.
    # File-list queries such as "list all spreadsheets" ask about
    # files on disk — they must NOT be treated as structured data
    # queries inside a spreadsheet workbook.
    active_ext = os.path.splitext(active_file)[1].lower() if active_file else ""
    is_spreadsheet_file = active_ext in {".xlsx", ".xls", ".csv", ".ods"}
    is_spreadsheet_scope = (modality == "spreadsheet" or is_spreadsheet_file)

    is_structured_spreadsheet_query = (
        not is_file_list_query
        and not is_count_temporal_query
        and (
            is_spreadsheet_scope
            or (
                has_spreadsheet_context
                and (has_spreadsheet_operation or has_spreadsheet_field)
            )
            or (
                has_spreadsheet_operation
                and (has_spreadsheet_field or modality == "spreadsheet")
            )
        )
    )


    # SOURCE REFERENCE DETECTION + SOURCE RESOLUTION
    # ============================================================

    source_hint = None
    source_confidence = 0.0
    unresolved_explicit_source = False
    is_ambiguous_source = False
    ambiguous_candidate_sources = []

    # ------------------------------------------------------------
    # 1. Explicit filename detection
    # ------------------------------------------------------------

    file_reference_pattern = re.compile(
        r"\b[\w.\-]+\."
        r"(?:jpg|jpeg|png|webp|gif|"
        r"pdf|doc|docx|xls|xlsx|csv|txt|"
        r"ppt|pptx)\b",
        re.IGNORECASE
    )

    file_reference_match = file_reference_pattern.search(question_lower)

    explicit_filename_found = bool(file_reference_match)

    if explicit_filename_found:
        matched_filename = file_reference_match.group(0)

        for fname in indexed_files:
            if fname.lower() == matched_filename.lower():
                source_hint = fname
                source_confidence = 1.0
                break

        # Preserve unresolved explicit filename.
        # Do NOT replace it with another file.
        if source_hint is None:
            source_hint = matched_filename
            source_confidence = 0.9
            unresolved_explicit_source = True


    # ------------------------------------------------------------
    # 2. Detect meaningful source/entity tokens
    # ------------------------------------------------------------

    entity_tokens = extract_entity_tokens(question_lower)

    meaningful_source_tokens = [
        token
        for token in entity_tokens
        if (
            token not in GENERIC_MEDIA_TERMS
            and token not in STOP_WORDS_SET
        )
    ]

    meaningful_source_name_found = bool(meaningful_source_tokens)


    # ------------------------------------------------------------
    # 3. Detect conversational/anaphoric source references
    # ------------------------------------------------------------

    anaphora_indicators = [
        r"\bit\b",
        r"\bthis\s+file\b",
        r"\bthat\s+file\b",
        r"\bthis\s+document\b",
        r"\bthat\s+document\b",
        r"\bthis\s+image\b",
        r"\bthat\s+image\b",
        r"\bthis\s+audio\b",
        r"\bthat\s+audio\b",
        r"\bthis\s+recording\b",
        r"\bthat\s+recording\b",
        r"\bthis\s+one\b",
        r"\bthat\s+one\b",
    ]

    has_anaphora = any(
        re.search(pattern, question_lower)
        for pattern in anaphora_indicators
    )


    # ------------------------------------------------------------
    # 4. Resolve anaphora only from previous conversation state
    # ------------------------------------------------------------

    if has_anaphora and source_hint is None:
        try:
            from app.services.interaction_state import get_last_interaction

            last_state = get_last_interaction()
            previous_source = None

            if last_state:
                previous_source = (
                    last_state.get("previous_source")
                    or last_state.get("canonical_source_id")
                )

                if not previous_source:
                    retrieved_sources = last_state.get("retrieved_sources")
                    if retrieved_sources:
                        previous_source = retrieved_sources[0]

            if previous_source:
                previous_basename = os.path.basename(
                    str(previous_source)
                )

                for fname in indexed_files:
                    if fname.lower() == previous_basename.lower():
                        source_hint = fname
                        source_confidence = 1.0
                        break

                if source_hint is None:
                    canonical_previous = resolve_canonical_source_id(
                        previous_basename
                    )

                    if canonical_previous:
                        source_hint = previous_basename
                        source_confidence = 0.95

        except Exception as exc:
            logger.warning(
                "Failed to resolve conversational source reference: %s",
                exc
            )


    # ------------------------------------------------------------
    # 5. Structural source-reference detection
    # ------------------------------------------------------------

    query_tokens = normalize_string(question_lower).split()

    source_reference_terms = {
        "image",
        "images",
        "photo",
        "photos",
        "picture",
        "pictures",
        "document",
        "documents",
        "notes",
        "note",
        "paper",
        "papers",
        "pdf",
        "spreadsheet",
        "spreadsheets",
        "sheet",
        "sheets",
        "file",
        "files",
        "call",
        "calls",
    }

    has_user_source_reference = False

    for index, token in enumerate(query_tokens):

        if token not in source_reference_terms:
            continue

        # Pattern:
        #   SQL notes
        #   mummy image
        #   Behari Lal call
        #
        # The token immediately before the media/source category
        # must be a meaningful entity.

        if index > 0:
            previous_token = query_tokens[index - 1]

            if (
                previous_token not in STOP_WORDS_SET
                and previous_token not in source_reference_terms
                and len(previous_token) >= 2
            ):
                has_user_source_reference = True
                break

        # Pattern:
        #   audio of jethlal
        #   recording from Behari
        #   image about mummy

        if index + 2 < len(query_tokens):
            relation = query_tokens[index + 1]
            following_token = query_tokens[index + 2]

            if (
                relation in {"of", "from", "about", "regarding"}
                and following_token not in STOP_WORDS_SET
                and following_token not in source_reference_terms
                and len(following_token) >= 2
            ):
                has_user_source_reference = True
                break


    # A meaningful query token is NOT automatically a source reference.
    #
    # IMPORTANT:
    # Words such as "education", "technology", "requirements",
    # "project", "conclusion", etc. can be content topics rather than
    # filenames. They must remain part of the user's question unless
    # they actually resolve to an indexed source.
    #
    # A token becomes a source reference only when it can be matched
    # against an indexed filename/stem/entity with sufficient confidence.
    #
    # Structured spreadsheet queries are handled separately and must
    # never use entity tokens as source references.

    # ------------------------------------------------------------
    # Source Resolution (PART A: Explicit Source First)
    # ------------------------------------------------------------
    # If the user explicitly mentions a filename or recognizable filename stem
    # (e.g. "from IT_Direct_Hire_companies", "in linux.pdf"), resolve that source
    # BEFORE deciding the final modality or intent handler.
    if source_hint is None and meaningful_source_name_found:
        src_match, src_conf = find_best_matching_source(
            question_lower,
            indexed_files,
            query_modality="all"
        )
        if isinstance(src_match, list):
            is_ambiguous_source = True
            ambiguous_candidate_sources = src_match
            source_confidence = src_conf
            has_user_source_reference = True
        elif src_match and src_conf >= 0.70:
            source_hint = src_match
            source_confidence = src_conf
            has_user_source_reference = True

    if source_hint is not None:
        resolved_ext = os.path.splitext(source_hint)[1].lower()
        if resolved_ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}:
            modality = "image"
        elif resolved_ext in {".xls", ".xlsx", ".csv", ".ods"}:
            modality = "spreadsheet"
        elif resolved_ext == ".pdf":
            modality = "pdf"
        elif resolved_ext in {".doc", ".docx", ".txt"}:
            modality = "docx"

    # Re-evaluate spreadsheet scope after resolving source
    active_ext = os.path.splitext(active_file)[1].lower() if active_file else ""
    hint_ext = os.path.splitext(source_hint)[1].lower() if source_hint else ""
    is_spreadsheet_file = active_ext in {".xlsx", ".xls", ".csv", ".ods"} or hint_ext in {".xlsx", ".xls", ".csv", ".ods"}
    is_spreadsheet_scope = (modality == "spreadsheet" or is_spreadsheet_file)

    is_structured_spreadsheet_query = (
        not is_file_list_query
        and not is_count_temporal_query
        and (
            is_spreadsheet_scope
            or (
                has_spreadsheet_context
                and (has_spreadsheet_operation or has_spreadsheet_field)
            )
            or (
                has_spreadsheet_operation
                and (has_spreadsheet_field or modality == "spreadsheet")
            )
            or (
                has_spreadsheet_field
                and modality == "spreadsheet"
            )
        )
    )

    if (
        meaningful_source_name_found
        and not is_structured_spreadsheet_query
        and source_hint is None
        and not explicit_filename_found
        and not has_anaphora
    ):
        source_match, source_match_confidence = find_best_matching_source(
            question_lower,
            indexed_files,
            query_modality=modality
        )

        if isinstance(source_match, list):
            has_user_source_reference = True
        elif (
            source_match is not None
            and source_match_confidence >= 0.70
        ):
            has_user_source_reference = True
        else:
            has_user_source_reference = False

    has_source_reference = (
        has_user_source_reference
        or has_anaphora
        or explicit_filename_found
    )


    # ------------------------------------------------------------
    # 6. Deterministic + semantic source resolution
    # ------------------------------------------------------------
    #
    # CRITICAL:
    # Never run source resolution for a generic query.
    #
    # "summarize the call"
    # "what was discussed in the recording"
    # "summarize the image"
    #
    # must NOT select an arbitrary indexed file.

    if (
        has_source_reference
        and source_hint is None
        and not unresolved_explicit_source
        and not has_anaphora
        and temporal_intent == "none"
    ):
        try:

            source_match, confidence = find_best_matching_source(
                question_lower,
                indexed_files,
                query_modality=modality
            )

            if isinstance(source_match, list):

                is_ambiguous_source = True
                ambiguous_candidate_sources = source_match
                source_confidence = confidence

            elif source_match and confidence >= 0.70:

                source_hint = source_match
                source_confidence = confidence

            else:

                top_src, top_score, second_score, candidate_list = (
                    resolve_semantic_source(
                        question_lower,
                        indexed_files,
                        query_modality=modality
                    )
                )

                # Strong unique semantic match
                if (
                    top_src
                    and top_score >= 0.55
                    and (top_score - second_score) >= 0.12
                ):
                    source_hint = top_src
                    source_confidence = top_score

                # Ambiguous semantic match
                elif (
                    top_src
                    and top_score >= 0.45
                    and len(candidate_list) > 1
                    and (top_score - second_score) < 0.12
                ):
                    is_ambiguous_source = True
                    ambiguous_candidate_sources = candidate_list

        except Exception as exc:
            logger.warning(
                "Source resolution failed: %s",
                exc
            )


    # ------------------------------------------------------------
    # 7. Explicit/meaningful source not resolved
    # ------------------------------------------------------------

    if (
        has_user_source_reference
        and source_hint is None
        and not is_ambiguous_source
        and explicit_filename_found
    ):
        if meaningful_source_tokens:

            unresolved_entity = meaningful_source_tokens[0]

            
            if modality == "image":
                source_hint = "UNRESOLVED_IMAGE_SOURCE"

            else:
                source_hint = (
                    f"UNRESOLVED_SOURCE_{unresolved_entity.upper()}"
                )

            unresolved_explicit_source = True


    # ------------------------------------------------------------
    # 8. Generic query safety
    # ------------------------------------------------------------

    if not has_source_reference:
        source_hint = None
        source_confidence = 0.0
        is_ambiguous_source = False
        ambiguous_candidate_sources = []
    # Incomplete / Ambiguous Source Detection
    if source_hint is None and temporal_intent == "none":
        incomplete_patterns = [
            r"\b(?:of|in|about|from|for|on|with|to|the|this|that|a|an)\s*$",
            r"\b(?:summarize|summarise|overview|synopsis|translate|read|describe|contents\s+of)\s+(?:the|this|that|a|an)\s*$"
        ]
        if any(re.search(pat, question_lower) for pat in incomplete_patterns) or (has_anaphora and not source_hint):
            is_ambiguous_source = True

    # Update modality if source_hint has explicit file extension
    if source_hint and not str(source_hint).startswith("UNRESOLVED_"):
        extension = os.path.splitext(source_hint)[1].lower()
        
        if extension in [".jpg", ".jpeg", ".png", ".webp"]:
            modality = "image"
        elif extension == ".pdf":
                modality = "pdf"
        elif extension in [".doc", ".docx"]:
            modality = "docx"
        elif extension in [".xls", ".xlsx", ".csv", ".ods"]:
            modality = "spreadsheet"
        elif extension in [".txt", ".md"]:
            modality = "text"    

    canonical_source_id = resolve_canonical_source_id(source_hint)

    # Step 5: Detect Intent Category
    image_name_query_patterns = [
        r"\bnames?\s+of\s+.*images?\b",
        r"\blist\s+.*name.*images?\b",
        r"\blist\s+.*images?.*name\b",
        r"\blist\s+.*images?.*named\b",
        r"\bimages?\s+which\s+have\s+named\b",
        r"\bfilenames?\s+of\b"
    ]
    is_image_filename_query = any(re.search(pat, question_lower) for pat in image_name_query_patterns)

    image_count_query_patterns = [
        r"\bhow\s+many\s+images?\s+.*named\b",
        r"\bhow\s+many\s+images?\s+.*by\s+the\s+names?\b",
        r"\bcount\s+of\s+images?\s+named\b"
    ]
    # ============================================================
    # GENERIC FILE COUNT QUERY
    # ============================================================
    # Detect what TYPE of files the user is asking to count.
    # The file type must occur immediately after the count phrase,
    # so queries such as "how many pages are in this PDF?" are NOT
    # mistaken for file-count queries.

    file_count_patterns = [
        # how many images / files / audio / videos ...
        r"\bhow\s+many\s+(?:image|images|photo|photos|picture|pictures|audio|audios|recording|recordings|video|videos|file|files|document|documents|pdf|pdfs|doc|docs|docx|spreadsheet|spreadsheets|excel|xlsx|csv|txt)\b",

        # number of images / files / audio / videos ...
        r"\bnumber\s+of\s+(?:image|images|photo|photos|picture|pictures|audio|audios|recording|recordings|video|videos|file|files|document|documents|pdf|pdfs|doc|docs|docx|spreadsheet|spreadsheets|excel|xlsx|csv|txt)\b",

        # count of images / files / audio / videos ...
        r"\bcount\s+of\s+(?:image|images|photo|photos|picture|pictures|audio|audios|recording|recordings|video|videos|file|files|document|documents|pdf|pdfs|doc|docs|docx|spreadsheet|spreadsheets|excel|xlsx|csv|txt)\b",

        # total number/count of ...
        r"\btotal\s+(?:number|count)\s+of\s+(?:image|images|photo|photos|picture|pictures|audio|audios|recording|recordings|video|videos|file|files|document|documents|pdf|pdfs|doc|docs|docx|spreadsheet|spreadsheets|excel|xlsx|csv|txt)\b",
    ]

    is_generic_file_count_query = any(
        re.search(pattern, question_lower, re.IGNORECASE)
        for pattern in file_count_patterns
    )

    # Folder-wide file count queries do not have a source file.
    # Prevent file-type words such as "videos" or "docx" from
    # being treated as unresolved filenames.
    if is_generic_file_count_query:
        source_hint = None
        source_confidence = 0.0
        unresolved_explicit_source = False

    chunk_metadata_patterns = [
        r"\bhow\s+many\s+chunks\b",
        r"\bchunk\s+count\b",
        r"\bwhen\s+was\s+this\s+file\s+added\b"
    ]
    is_chunk_metadata_query = any(re.search(pat, question_lower) for pat in chunk_metadata_patterns)

    
    # Detect IMAGE_DISPLAY intent: user wants to retrieve/view/see an image, not describe it.
    # Uses structural patterns (action verb + visual object) rather than a synonym list.
    # Intentionally kept separate from VISUAL_QA (describe/analyse) and SUMMARIZATION.
    image_display_patterns = [
        r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\bdisplay\s+(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\bopen\s+(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\blet\s+me\s+see\s+(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\bi\s+want\s+to\s+see\s+(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\bcan\s+(?:you\s+)?show\s+(?:me\s+)?(?:the\s+)?(?:image|photo|picture|pic)\b",
        r"\bfetch\s+(?:the\s+)?(?:image|photo|picture|pic)\b",
    ]
    # Image display is only triggered when the modality resolves to image (avoids false positives
    # on queries like "show me the audio" which belong to a different handler).
    is_image_display_query = (
        modality == "image"
        and any(re.search(pat, question_lower) for pat in image_display_patterns)
    )

    ocr_phrases = [
        "text in the image", "text in image", "what is the text", "read text",
        "ocr text", "words in the image", "writing in the image", "written in",
        "what is written", "text of image",
        "all the names in", "all names in", "all the text in", "all text in",
        "all the text in", "all text in", "text from", "content of", "content of dense image",
        "content of dense"
    ]
    is_ocr_query = any(op in question_lower for op in ocr_phrases)

    image_summary_phrases = [
        "summarize the image",
        "summarise the image",
        "summary of the image",
        "summarize image",
        "summarise image",
        "describe the image",
        "describe image",
        "give me a summary of the image",
        "provide a summary of the image"
    ]
    is_image_summary_query = any(isp in question_lower for isp in image_summary_phrases)

    # Generic detection for full-source transcript, complete content, and full-source translation requests
    full_source_patterns = [
        r"\ball\s+(?:the\s+)?text\b", r"\bcomplete\s+(?:the\s+)?transcript\b",
        r"\bcomplete\s+(?:the\s+)?text\b", r"\bfull\s+text\b", r"\bcomplete\s+document\b",
        r"\bfull\s+document\b", r"\bentire\s+text\b", r"\bentire\s+document\b",
        r"\bentire\s+transcript\b", r"\bfull\s+transcript\b", r"\bentire\s+audio\b",
        r"\ball\s+of\s+the\s+audio\b", r"\bcomplete\s+audio\b", r"\bfull\s+content\b",
        r"\bcomplete\s+content\b", r"\beverything\s+said\b", r"\beverything\s+in\b",
        r"\beverything\s+from\b", r"\ball\s+(?:the\s+)?chapters?\b", r"\blist\s+all\s+chapters?\b",
        r"\ball\s+(?:the\s+)?sections?\b", r"\ball\s+(?:the\s+)?pages?\b", r"\bentire\s+book\b",
        r"\bentire\s+file\b", r"\bentire\s+paper\b", r"\bwhole\s+text\b", r"\bwhole\s+transcript\b",
        r"\bwhole\s+book\b", r"\bwhole\s+file\b", r"\bconvert\s+all\b", r"\btranslate\s+all\b",
        r"\btranslate\s+(?:the\s+)?entire\b", r"\bconvert\s+(?:the\s+)?entire\b",
        r"\btranslate\s+(?:the\s+)?whole\b", r"\bconvert\s+(?:the\s+)?whole\b",
        r"\btranslate\s+everything\b", r"\bconvert\s+everything\b",
        r"\btranslate\s+(?:the\s+)?complete\b", r"\bconvert\s+(?:the\s+)?complete\b",
        r"\bcomplete\s+english\s+version\b", r"\bfull\s+english\s+version\b",
        r"\bcomplete\s+translation\b", r"\bfull\s+translation\b",
        r"\ball\s+(?:the\s+)?spoken\s+content\b", r"\beverything\s+spoken\b",
        r"\bfull\s+spoken\s+transcript\b", r"\ball\s+the\s+audio\s+content\b",
        r"\ball\s+audio\s+content\b", r"\btranslate\s+every\s+spoken\s+line\b",
        r"\bpoore\s+audio\b", r"\bcomplete\s+transcript\s+do\b", r"\bis\s+call\s+ka\s+complete\s+transcript\b",
        r"\baudio\s+ko\s+english\s+me\s+convert\b", r"\bwithout\s+summarizing\b", r"\bwithout\s+summarising\b"
    ]
    is_full_source_request = any(re.search(pattern_string, question_lower) for pattern_string in full_source_patterns)

    # Detect request scope generically
    partial_topic_patterns = [
        r"\bpart\b", r"\bsection\b", r"\btopic\b", r"\bsegment\b", r"\bportion\b",
        r"\babout\b", r"\bregarding\b", r"\brelated\s+to\b", r"\bdiscussing\b"
    ]
    has_partial_topic_indicator = any(re.search(pat, question_lower) for pat in partial_topic_patterns)

    # Detect explicit target language if requested in query
    target_language = extract_target_language(question_lower)

    # Detect translation/conversion action verbs
    translation_action_patterns = [
        r"\bconvert\b", r"\btranslate\b", r"\btranslation\b", r"\bconversion\b"
    ]
    is_translation_action = any(re.search(pattern_string, question_lower) for pattern_string in translation_action_patterns)

    is_full_translation_request = False
    if is_full_source_request and (is_translation_action or target_language is not None):
        is_full_translation_request = True
    elif (
        is_translation_action
        and target_language is not None
        and not has_partial_topic_indicator
        and source_hint is not None
    ):
        is_full_translation_request = True

    if is_full_source_request or is_full_translation_request:
        request_scope = "complete_file"
    elif has_partial_topic_indicator:
        request_scope = "partial_topic"
    else:
        request_scope = "selective"

    # ------------------------------------------------------------
    # Structured spreadsheet query detection
    # ------------------------------------------------------------
    # These queries must bypass semantic RAG because they operate on
    # complete spreadsheet rows rather than semantically similar chunks.
    #
    # Examples:
    #   list all product based companies
    #   show backend companies in Bangalore
    #   sort companies by CTC
    #   sort CTC of each city
    #   product companies in Koramangala
    #   give company name and package
    #
    # This is intentionally query-pattern based, not workbook/file-name
    # specific. The actual workbook is resolved by the spreadsheet handler.

    # Intent priority order:
    # Correction > Metadata/Temporal > Image Display > Image Metadata > Speaker >
    # Full Content / Translation > OCR > Image Summary > Audio Summary > Document Summary > Text Search
    source_type_patterns = [
        r"^\s*what\s+type\s+is\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*what\s+kind\s+of\s+file\s+is\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*what\s+kind\s+of\s+file\s+is\s+the\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*is\s+(.+?)\s+(an?|the)\s+(image|audio|video|pdf|document|file)\s*[?.!]*\s*$",
    ]

    is_source_type_query = (
        bool(source_hint)
        and any(
            re.match(pattern, question_lower, re.IGNORECASE)
            for pattern in source_type_patterns
        )
        and not (
            modality == "image"
            and canonical_source_id is not None
            and re.match(
                r"^\s*what\s+is\s+.+?\s*[?.!]*\s*$",
                question_lower,
                re.IGNORECASE
            )
        )
    )




    if is_correction:
        intent = INTENT_CORRECTION
    elif is_generic_file_count_query:
        # "How many spreadsheets/PDFs/images do I have?" — counts files on disk.
        # Must win over is_structured_spreadsheet_query, which only handles
        # data queries INSIDE a workbook.
        intent = INTENT_FILE_COUNT
    elif is_file_list_query and start_dt is None:
        # Plain file-list query with NO temporal constraint.
        # Examples: "list all files", "list all spreadsheets", "show all PDFs".
        # These enumerate the authoritative file inventory without any date filter.
        intent = INTENT_FILE_LIST
    elif is_structured_spreadsheet_query:
        intent = INTENT_SPREADSHEET_QUERY
    elif temporal_intent != "none":
        # File-list OR count query that includes a date range filter.
        # Examples: "files uploaded yesterday", "which files were added today".
        intent = INTENT_TEMPORAL_FILE_QUERY
    elif is_image_display_query:
        # User wants to see/retrieve an image, not analyse it.
        # Placed before image_filename and image_summary to avoid misclassification.
        intent = "IMAGE_DISPLAY"
    elif is_image_filename_query:
        intent = INTENT_IMAGE_FILENAME_QUERY
    elif is_chunk_metadata_query:
        intent = INTENT_FILE_METADATA
    elif is_source_type_query:
        intent = INTENT_SOURCE_TYPE
    elif is_full_translation_request:
        intent = "FULL_CONTENT_FETCH"
    elif is_full_source_request:
        intent = "FULL_CONTENT_FETCH"
    elif is_ocr_query:
        intent = INTENT_IMAGE_OCR
    elif is_image_summary_query:
        intent = INTENT_IMAGE_SUMMARY
    elif (
        modality == "image"
        and not is_image_display_query
        and not is_image_filename_query
        and not is_count_temporal_query
        and not is_chunk_metadata_query
    ):
        # Route natural-language question in image mode to visual QA
        intent = INTENT_IMAGE_VISUAL_QUERY
    
    elif any(word in question_lower for word in ["summarize", "summarise", "summary", "overview"]):
        if modality == "image":
            intent = INTENT_IMAGE_SUMMARY
        else:
            intent = INTENT_DOCUMENT_SUMMARY
    else:
        intent = INTENT_TEXT_SEARCH

    visual_qa_keywords = [
        "what is in", "what's in", "content of", "describe", "what color", "how many",
        "what text", "read text", "explain image", "what does it look like"
    ]
    is_visual_qa = (
        (modality == "image" and any(kw in question_lower for kw in visual_qa_keywords)) or
        is_image_summary_query or
        is_ocr_query
    )

    # ------------------------------------------------------------
    # Face query classification (CONTEXT-GATED)
    # ------------------------------------------------------------
    # Face-specific interpretation belongs to Query Understanding.
    # Handlers consume this structured result instead of reparsing
    # the user's natural-language question.
    face_intent = FaceIntent.NONE
    face_person_name = None

    active_ext_face = os.path.splitext(active_file)[1].lower() if active_file else ""
    is_image_file_context = active_ext_face in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    is_image_modality_context = (modality == "image")
    has_explicit_visual_keywords = bool(re.search(
        r"\b(?:face|faces|photo|photos|picture|pictures|image|images|person\s+in\s+(?:the\s+)?(?:image|photo|picture)|this\s+person|in\s+this\s+photo|in\s+this\s+image)\b",
        question_lower
    ))
    has_visual_context = is_image_file_context or is_image_modality_context or has_explicit_visual_keywords

    if has_visual_context and not is_structured_spreadsheet_query:
        face_count_patterns = (
            r"\bhow\s+many\s+(?:people|persons|person|faces)\b",
            r"\bnumber\s+of\s+(?:people|persons|person|faces)\b",
            r"\bcount\s+(?:the\s+)?(?:people|persons|person|faces)\b",
        )

        face_presence_patterns = (
            r"\bis\s+there\s+(?:a\s+)?person\b",
            r"\bis\s+there\s+anyone\b",
            r"\bare\s+there\s+people\b",
            r"\bis\s+anyone\b",
            r"\bdoes\s+the\s+image\s+contain\s+(?:a\s+)?person\b",
            r"\bdoes\s+the\s+image\s+contain\s+people\b",
        )

        face_identity_patterns = (
            r"\bwho\s+is\b",
            r"\bwho\s+are\b",
            r"\bwhose\s+face\b",
            r"\bidentify\b",
            r"\brecognize\b",
            r"\bwhich\s+person\b",
            r"\bis\s+this\s+person\b",
        )

        if any(re.search(pattern, question_lower) for pattern in face_count_patterns):
            face_intent = FaceIntent.COUNT

        elif any(re.search(pattern, question_lower) for pattern in face_presence_patterns):
            face_intent = FaceIntent.PRESENCE

        else:
            face_person_name = extract_person_name_from_question(question)

            # A person name does not by itself mean identification.
            # Explicit identity questions identify a face; image-search
            # phrasing searches the persistent face memory across images.
            if any(
                re.search(pattern, question_lower)
                for pattern in face_identity_patterns
            ):
                face_intent = FaceIntent.IDENTIFICATION

            elif is_face_search_query(question):
                face_intent = FaceIntent.SEARCH

    if face_intent != FaceIntent.NONE and not is_structured_spreadsheet_query:
        intent = INTENT_IMAGE_FACE_QUERY

    analysis_result = {
        "intent": intent,
        "request_scope": request_scope,
        "temporal_intent": temporal_intent,
        "face_intent": face_intent.value,
        "face_person_name": face_person_name,
        "is_visual_qa": is_visual_qa,
        "is_ocr_query": is_ocr_query,
        "is_image_summary_query": is_image_summary_query,
        "modality": modality,
        "source_hint": source_hint,
        "canonical_source_id": canonical_source_id,
        "unresolved_explicit_source": unresolved_explicit_source,
        "source_confidence": source_confidence,
        "target_language": target_language,
        "is_correction": is_correction,
        "start_datetime": start_dt,
        "end_datetime": end_dt,
        "date_label": date_label,
        "extracted_limit": extracted_limit,
        "needs_database": True
    }

    # Construct strongly-typed QueryPlan object
    modality_enum_map = {
        "image": Modality.IMAGE,
        "document": Modality.DOCUMENT,
        "pdf": Modality.PDF,
        "docx": Modality.DOCX,
        "spreadsheet": Modality.SPREADSHEET,
        "all": Modality.ALL
    }
    target_modality_enum = modality_enum_map.get(modality, Modality.ALL)

    intent_enum_map = {
        INTENT_CORRECTION: QueryIntent.CORRECTION,
        INTENT_SPREADSHEET_QUERY: QueryIntent.SPREADSHEET_QUERY,
        INTENT_SOURCE_TYPE: QueryIntent.SOURCE_TYPE,
        INTENT_FILE_METADATA: QueryIntent.METADATA_QUERY,
        INTENT_FILE_COUNT: QueryIntent.METADATA_QUERY,
        INTENT_FILE_LIST: QueryIntent.METADATA_QUERY,
        INTENT_TEMPORAL_FILE_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_IMAGE_FILENAME_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_IMAGE_COUNT_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_IMAGE_FACE_QUERY: QueryIntent.IMAGE_FACE_QUERY,
        # IMAGE_DISPLAY is registered as a plain string because it is a new enum value.
        # This mapping is the single place where it resolves to QueryIntent.IMAGE_DISPLAY.
        "IMAGE_DISPLAY": QueryIntent.IMAGE_DISPLAY,
        INTENT_IMAGE_OCR: QueryIntent.VISUAL_QA,
        INTENT_IMAGE_VISUAL_QUERY: QueryIntent.VISUAL_QA,
        INTENT_IMAGE_SUMMARY: QueryIntent.VISUAL_QA if modality != "image" else QueryIntent.SUMMARIZATION,
        INTENT_DOCUMENT_SUMMARY: QueryIntent.SUMMARIZATION,
        INTENT_TEXT_SEARCH: QueryIntent.QUESTION_ANSWERING,
        "FULL_CONTENT_FETCH": QueryIntent.FULL_CONTENT_FETCH,
    }
    

    if request_scope == "complete_file":
        target_intent_enum = QueryIntent.FULL_CONTENT_FETCH
        target_scope_enum = RequestScope.COMPLETE_FILE
    elif (
        intent == INTENT_DOCUMENT_SUMMARY
        and not has_partial_topic_indicator
    ):
        target_intent_enum = QueryIntent.SUMMARIZATION
        target_scope_enum = RequestScope.SUMMARY
    else:
        target_intent_enum = intent_enum_map.get(
            intent,
            QueryIntent.QUESTION_ANSWERING
        )

        scope_enum_map = {
            "complete_file": RequestScope.COMPLETE_FILE,
            "summary": RequestScope.SUMMARY,
            "selective": RequestScope.QUESTION_ANSWER,
            "partial_topic": RequestScope.QUESTION_ANSWER
        }

        if target_intent_enum == QueryIntent.METADATA_QUERY:
            target_scope_enum = RequestScope.METADATA_ONLY
        elif target_intent_enum == QueryIntent.SUMMARIZATION:
            target_scope_enum = RequestScope.SUMMARY
        else:
            target_scope_enum = scope_enum_map.get(
                request_scope,
                RequestScope.QUESTION_ANSWER
            )
    # Folder-wide file-count queries never refer to a specific source file.
    # File-type words such as "videos", "docx", "pdf", etc. describe the
    # requested category, not a filename.
    if intent == "FILE_COUNT":
        source_is_explicit = False
        unresolved_explicit_source = False
        source_hint = None
        canonical_source_id = None
        source_confidence = 0.0
    else:
        source_is_explicit = bool(
            has_user_source_reference
            or explicit_filename_found
            or unresolved_explicit_source
        )
    
    # Check if the resolved source actually exists in the database index or filesystem
    indexed_lower_files = [f.lower() for f in indexed_files] if indexed_files else []
    resolved_file_exists = False
    if canonical_source_id and not str(canonical_source_id).startswith("UNRESOLVED_"):
        base_name = os.path.basename(canonical_source_id).lower()
        if base_name in indexed_lower_files or os.path.exists(canonical_source_id):
            resolved_file_exists = True

    source_is_resolved = resolved_file_exists and not unresolved_explicit_source

    query_operation = infer_query_operation(
        question=question,
        intent=intent,
        request_scope=request_scope,
    )

    # Resolve the requested file category once during query understanding.
    # Handlers should consume QueryPlan.file_type instead of reparsing
    # the user's raw question.
    file_type = None

    if intent in {INTENT_FILE_COUNT, INTENT_IMAGE_COUNT_QUERY}:
        if any(word in question_lower for word in (
            "image", "images", "photo", "photos",
            "picture", "pictures"
        )):
            file_type = "image"
        elif re.search(r"\bpdfs?\b", question_lower):
            file_type = "pdf"
        elif any(word in question_lower for word in (
            "document", "documents", "doc", "docs"
        )):
            file_type = "document"
        elif any(word in question_lower for word in (
            "spreadsheet", "spreadsheets", "excel", "xlsx"
        )):
            file_type = "spreadsheet"
        elif any(word in question_lower for word in (
            "presentation", "presentations",
            "powerpoint", "powerpoint files",
            "ppt", "pptx"
        )):
            file_type = "presentation"
        elif any(word in question_lower for word in (
            "archive", "archives", "zip", "tar"
        )):
            file_type = "archive"
        else:
            file_type = "all"

    plan = QueryPlan(
        raw_query=question,
        normalized_query=question_lower,
        intent=target_intent_enum,
        operation=query_operation,
        scope=target_scope_enum,
        modality=target_modality_enum,
        file_type=file_type,
        face_intent=face_intent,
        face_person_name=face_person_name,
        source_spec=SourceSpec(
            source_hint=source_hint,
            canonical_path=canonical_source_id if resolved_file_exists else None,
            is_explicit=source_is_explicit,
            is_resolved=source_is_resolved,
            is_ambiguous=is_ambiguous_source,
            confidence=source_confidence if resolved_file_exists else 0.0,
            candidate_sources=ambiguous_candidate_sources
        ),
        filters=QueryFilters(
            start_datetime=start_dt,
            end_datetime=end_dt,
            date_label=date_label,
            extracted_limit=extracted_limit,
            target_language=target_language
        ),
        is_correction=is_correction,
        metadata={"is_ocr_query": is_ocr_query, "is_image_summary_query": is_image_summary_query}
    )

    analysis_result["plan"] = plan
    return analysis_result


def build_query_plan(question, indexed_files=None): #-> QueryPlan
    """
    Helper function to directly analyze a question and return a strongly-typed QueryPlan.
    """
    analysis = analyze_query(question, indexed_files=indexed_files)
    return analysis["plan"]