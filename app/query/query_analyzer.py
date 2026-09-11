import re
import os
import unicodedata
import difflib
from app.storage.lancedb_store import get_hash_table
from app.query.query_plan import QueryPlan, QueryIntent, RequestScope, Modality, SourceSpec, QueryFilters

# ============================================================
# QUERY INTENT TAXONOMY CONSTANTS
# ============================================================
INTENT_SOURCE_LOOKUP = "SOURCE_LOOKUP"
INTENT_FILE_METADATA = "FILE_METADATA"
INTENT_FILE_COUNT = "FILE_COUNT"
INTENT_FILE_LIST = "FILE_LIST"
INTENT_TEMPORAL_FILE_QUERY = "TEMPORAL_FILE_QUERY"
INTENT_TEXT_SEARCH = "TEXT_SEARCH"
INTENT_AUDIO_SEARCH = "AUDIO_SEARCH"
INTENT_AUDIO_SUMMARY = "AUDIO_SUMMARY"
INTENT_AUDIO_TRANSCRIPT = "AUDIO_TRANSCRIPT"
INTENT_AUDIO_TRANSLATION = "AUDIO_TRANSLATION"
INTENT_AUDIO_SPEAKER_QUERY = "AUDIO_SPEAKER_QUERY"
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
    "audio", "image", "images", "video", "picture", "photo", "recording", "recordings",
    "sound", "file", "files", "document", "documents", "pdf", "docx", "xlsx", "mp3",
    "m4a", "wav", "mpeg", "txt", "excel", "list", "name", "names", "page", "pages",
    "transcript", "transcripts", "content", "contents", "summary", "summaries"
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
    "english", "hindi", "hinglish", "french", "german", "spanish", "italian", "portuguese", "japanese", "chinese", "russian"
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


def normalize_string(text_input):
    """
    Normalizes text by lowercasing, NFKD unicode normalization, stripping apostrophes/possessives,
    and replacing punctuation/symbols with single spaces.
    Example: "jethlal's" -> "jethlal", "Behari_lal-call.m4a" -> "behari lal call m4a"
    """
    if not text_input:
        return ""
    
    # 1. Unicode normalization (NFKD)
    normalized_unicode = unicodedata.normalize("NFKD", str(text_input))
    
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
    audio_extensions = (".m4a", ".mp3", ".wav", ".mpeg", ".aac", ".flac")

    # Filter candidate files by modality if specified
    candidate_files = []
    for indexed_file in indexed_files:
        filename_lower = indexed_file.lower()
        if query_modality == "image" or face_intent in ["face_search", "face_identification"]:
            if filename_lower.endswith(image_extensions):
                candidate_files.append(indexed_file)
        elif query_modality == "audio":
            if filename_lower.endswith(audio_extensions):
                candidate_files.append(indexed_file)
        else:
            candidate_files.append(indexed_file)

    if not candidate_files:
        candidate_files = indexed_files

    normalized_question = normalize_string(question_lower)
    compact_question = compact_alphanumeric(question_lower)

    # Stage 1: Exact Full Filename in Query
    for candidate in candidate_files:
        if candidate.lower() in question_lower:
            return candidate, 1.0

    # Stage 2: Entity Token Matching (Exact + Fuzzy Match)
    question_entity_tokens = extract_entity_tokens(question_lower)

    if question_entity_tokens:
        scored_candidates = []
        for candidate in candidate_files:
            file_stem = os.path.splitext(candidate)[0]
            stem_tokens = extract_entity_tokens(file_stem)

            best_token_score = 0.0
            exact_match_count = 0

            for q_token in question_entity_tokens:
                for s_token in stem_tokens:
                    if q_token == s_token:
                        exact_match_count += 1
                        best_token_score = max(best_token_score, 1.0)
                        break
                    
                    # Fuzzy match check for typos (e.g. jethlal vs jethalal)
                    similarity = calculate_token_similarity(q_token, s_token)
                    if len(q_token) >= 4 and len(s_token) >= 4:
                        if similarity >= 0.70:
                            best_token_score = max(best_token_score, similarity)
                    elif (len(q_token) >= 4 and q_token in s_token) or (len(s_token) >= 4 and s_token in q_token):
                        best_token_score = max(best_token_score, 0.85)

            if best_token_score >= 0.70:
                scored_candidates.append((best_token_score, exact_match_count, candidate))

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

    return source_hint


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


def analyze_query(question, indexed_files=None):
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
        r"\blist\s+(?:all\s+)?audio\b", r"\bshow\s+(?:all\s+)?audio\b", r"\blist\s+(?:all\s+)?documents?\b", r"\bshow\s+(?:all\s+)?documents?\b",
        r"\bwhich\s+files\b", r"\bwhich\s+audio\b", r"\bwhich\s+images?\b", r"\bwhich\b.*\b(?:files?|images?|audio|recordings?|documents?)\b",
        r"\bfiles?\s+uploaded\b", r"\bfiles?\s+added\b", r"\bimages?\s+added\b", r"\baudio\s+added\b", r"\brecordings?\s+added\b",
        r"\brecent\s+files?\b", r"\blatest\s+files?\b", r"\bnewest\s+files?\b", r"\bmost\s+recent\b",
        r"\brecently\s+added\b", r"\bnewly\s+added\b", r"\bdate-wise\b", r"\bdatewise\b", r"\bby\s+date\b",
        r"\b(?:top|latest|recent|first)\s+\d+\s*(?:files?|images?|audio|recordings?|documents?)\b",
        r"\b\d+\s+(?:recent|latest)\s+(?:files?|images?|audio|recordings?|documents?)\b",
        r"\bwhat\s+(?:audio\s+)?files\b", r"\bnames?\s+of\s+(?:my\s+)?(?:audio\s+)?files?\b",
        r"\brecordings?\s+in\s+(?:my\s+)?folder\b", r"\bshow\s+(?:me\s+)?(?:all\s+)?audio\s+recordings?\b",
        r"\bshow\s+(?:every\s+)?sound\s+file\b", r"\ball\s+audio\s+recordings?\b"
    ]
    is_file_list_query = any(re.search(pat, question_lower) for pat in file_list_patterns) or (start_dt is not None)

    count_temporal_patterns = [
        r"\bhow\s+many\s+files\b", r"\bhow\s+many\s+images\b", r"\bhow\s+many\s+audio\b", r"\bcount\s+of\s+files\b"
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
        r"\b(?:top|latest|recent|first)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:recent|latest|files?|images?|audio|documents?)\b",
        question_without_date_words
    )
    extracted_limit = None
    if explicit_limit_match:
        number_string = explicit_limit_match.group(1) or explicit_limit_match.group(2)
        if number_string:
            extracted_limit = int(number_string)

    # Default limit to 5 if user asked for recent/latest files without specifying an explicit limit
    if extracted_limit is None and any(word in question_lower for word in ["recent", "latest", "newest"]):
        extracted_limit = 5

    temporal_intent = "none"
    if is_count_temporal_query:
        temporal_intent = INTENT_FILE_COUNT
    elif is_file_list_query:
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY

    # Step 3: Modality & File Type Categorization
    modality = "all"
    audio_keywords = ["audio", "sound", "m4a", "mp3", "wav", "mpeg", "recording", "speaker", "speak", "voice", "con call", "talk"]
    image_keywords = ["image", "images", "picture", "pictures", "photo", "photos", "visual", "diagram", "chart", "figure", "jpg", "jpeg", "png", "webp"]
    pdf_keywords = ["pdf", "pdfs"]
    doc_keywords = ["docx", "doc", "document", "documents"]
    video_keywords = ["video", "videos", "mp4", "mkv", "avi", "mov"]

    if any(kw in question_lower for kw in audio_keywords):
        modality = "audio"
    elif any(kw in question_lower for kw in image_keywords):
        modality = "image"
    elif any(kw in question_lower for kw in pdf_keywords):
        modality = "pdf"
    elif any(kw in question_lower for kw in doc_keywords):
        modality = "docx"
    elif any(kw in question_lower for kw in video_keywords):
        modality = "video"

    # Step 4: Source Hint Resolution & Explicit Source Detection
    source_hint = None
    source_confidence = 0.0
    unresolved_explicit_source = False

    if temporal_intent == "none":
        # Check explicit file extensions
        file_match = re.search(r"\b([a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|m4a|mp3|wav|mpeg|pdf|docx|xlsx))\b", question_lower)
        if file_match:
            matched_str = file_match.group(1)
            for fname in indexed_files:
                if fname.lower() == matched_str.lower():
                    source_hint = fname
                    source_confidence = 1.0
                    break
            if source_hint is None:
                source_hint = matched_str
                source_confidence = 0.9

        if source_hint is None:
            source_match, confidence = find_best_matching_source(question_lower, indexed_files, query_modality=modality)
            if source_match and confidence >= 0.70:
                source_hint = source_match
                source_confidence = confidence

    # Detect if query has explicit source/entity reference that failed resolution
    entity_tokens = extract_entity_tokens(question_lower)
    source_phrase_indicators = ["audio of", "in file", "the file", "image of", "recording of", "file", "audio", "image"]
    has_source_phrase = any(ind in question_lower for ind in source_phrase_indicators)

    if source_hint is None and temporal_intent == "none":
        if has_source_phrase and len(entity_tokens) > 0:
            target_entity = entity_tokens[0]
            if modality == "audio" or "audio" in question_lower:
                source_hint = "UNRESOLVED_AUDIO_SOURCE"
            else:
                source_hint = f"UNRESOLVED_SOURCE_{target_entity.upper()}"
            unresolved_explicit_source = True

    # Update modality if source_hint has explicit file extension
    if source_hint and not str(source_hint).startswith("UNRESOLVED_"):
        extension = os.path.splitext(source_hint)[1].lower()
        if extension in [".m4a", ".mp3", ".wav", ".mpeg"]:
            modality = "audio"
        elif extension in [".jpg", ".jpeg", ".png", ".webp"]:
            modality = "image"

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
    is_image_count_query = any(re.search(pat, question_lower) for pat in image_count_query_patterns)

    chunk_metadata_patterns = [
        r"\bhow\s+many\s+chunks\b",
        r"\bchunk\s+count\b",
        r"\bwhen\s+was\s+this\s+file\s+added\b"
    ]
    is_chunk_metadata_query = any(re.search(pat, question_lower) for pat in chunk_metadata_patterns)

    speaker_patterns = [
        r"\bwho\s+is\s+speakers?\b",
        r"\bwho\s+are\s+the\s+speakers?\b",
        r"\bhow\s+many\s+persons?\s+are\s+talking\b",
        r"\bhow\s+many\s+people\s+are\s+talking\b",
        r"\bis\s+there\s+any\s+lady\s+talk\b",
        r"\bis\s+a\s+woman\s+speaking\b",
        r"\bwho\s+is\s+speaking\b",
        r"\bwho\s+speaks\b"
    ]
    is_speaker_query = any(re.search(pat, question_lower) for pat in speaker_patterns)

    ocr_phrases = [
        "text in the image", "text in image", "what is the text", "read text",
        "ocr text", "words in the image", "writing in the image", "text of image",
        "all the names in", "all names in", "all the text in", "all text in",
        "text from dense", "text from dense2", "text from dense3", "content of dense image",
        "content of dense"
    ]
    is_ocr_query = any(op in question_lower for op in ocr_phrases)

    image_summary_phrases = [
        "summarize the image", "summarise the image", "summary of the image",
        "summarize image", "summarise image", "describe the image", "describe image",
        "summarize dense", "summarize dense2", "summarize dense3"
    ]
    is_image_summary_query = any(isp in question_lower for isp in image_summary_phrases)

    # Generic detection for full-source transcript and full-source translation requests
    full_source_patterns = [
        r"\ball\s+(?:the\s+)?text\b", r"\bcomplete\s+(?:the\s+)?transcript\b",
        r"\bcomplete\s+(?:the\s+)?text\b", r"\bfull\s+text\b", r"\bcomplete\s+document\b",
        r"\bfull\s+document\b", r"\bentire\s+text\b", r"\bentire\s+document\b",
        r"\bentire\s+transcript\b", r"\bfull\s+transcript\b", r"\bentire\s+audio\b",
        r"\ball\s+of\s+the\s+audio\b", r"\bcomplete\s+audio\b", r"\bfull\s+content\b",
        r"\bcomplete\s+content\b", r"\beverything\s+said\b", r"\beverything\s+in\b",
        r"\bwhole\s+text\b", r"\bwhole\s+transcript\b", r"\bconvert\s+all\b", r"\btranslate\s+all\b",
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
    elif is_translation_action and target_language is not None and not has_partial_topic_indicator and (modality in ["audio", "document"] or source_hint is not None):
        is_full_translation_request = True

    if is_full_source_request or is_full_translation_request:
        request_scope = "complete_file"
    elif has_partial_topic_indicator:
        request_scope = "partial_topic"
    else:
        request_scope = "selective"

    audio_summary_phrases = [
        "summarize", "summarise", "summary", "overview",
        "kya baatein hui", "kya discuss", "kya baat hui", "kya hua",
        "summary do", "summary batao", "what was discussed", "what happened in",
        "discussion in", "details of call", "con call me"
    ]
    is_audio_summary_query = any(asp in question_lower for asp in audio_summary_phrases)

    if is_correction:
        intent = INTENT_CORRECTION
    elif temporal_intent != "none":
        intent = INTENT_TEMPORAL_FILE_QUERY
    elif is_image_filename_query:
        intent = INTENT_IMAGE_FILENAME_QUERY
    elif is_image_count_query:
        intent = INTENT_IMAGE_COUNT_QUERY
    elif is_chunk_metadata_query:
        intent = INTENT_FILE_METADATA
    elif is_speaker_query:
        intent = INTENT_AUDIO_SPEAKER_QUERY
    elif is_full_translation_request:
        intent = INTENT_AUDIO_TRANSLATION
    elif is_full_source_request:
        intent = INTENT_AUDIO_TRANSCRIPT
    elif is_ocr_query:
        intent = INTENT_IMAGE_OCR
    elif is_image_summary_query:
        intent = INTENT_IMAGE_SUMMARY
    elif is_audio_summary_query and modality == "audio":
        intent = INTENT_AUDIO_SUMMARY
    elif any(word in question_lower for word in ["summarize", "summarise", "summary", "overview"]):
        if modality == "audio":
            intent = INTENT_AUDIO_SUMMARY
        elif modality == "image":
            intent = INTENT_IMAGE_SUMMARY
        else:
            intent = INTENT_DOCUMENT_SUMMARY
    else:
        intent = INTENT_TEXT_SEARCH

    analysis_result = {
        "intent": intent,
        "request_scope": request_scope,
        "temporal_intent": temporal_intent,
        "face_intent": "none",
        "is_visual_qa": (modality == "image") or is_image_summary_query or is_ocr_query,
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
        "audio": Modality.AUDIO,
        "image": Modality.IMAGE,
        "document": Modality.DOCUMENT,
        "pdf": Modality.PDF,
        "docx": Modality.DOCX,
        "video": Modality.VIDEO,
        "all": Modality.ALL
    }
    target_modality_enum = modality_enum_map.get(modality, Modality.ALL)

    intent_enum_map = {
        INTENT_CORRECTION: QueryIntent.CORRECTION,
        INTENT_FILE_METADATA: QueryIntent.METADATA_QUERY,
        INTENT_FILE_COUNT: QueryIntent.METADATA_QUERY,
        INTENT_FILE_LIST: QueryIntent.METADATA_QUERY,
        INTENT_TEMPORAL_FILE_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_IMAGE_FILENAME_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_IMAGE_COUNT_QUERY: QueryIntent.METADATA_QUERY,
        INTENT_AUDIO_SPEAKER_QUERY: QueryIntent.SPEAKER_ANALYSIS,
        INTENT_AUDIO_TRANSLATION: QueryIntent.FULL_CONTENT_FETCH,
        INTENT_AUDIO_TRANSCRIPT: QueryIntent.FULL_CONTENT_FETCH,
        INTENT_IMAGE_OCR: QueryIntent.VISUAL_QA,
        INTENT_IMAGE_SUMMARY: QueryIntent.SUMMARIZATION if modality == "image" else QueryIntent.VISUAL_QA,
        INTENT_AUDIO_SUMMARY: QueryIntent.SUMMARIZATION,
        INTENT_DOCUMENT_SUMMARY: QueryIntent.SUMMARIZATION,
        INTENT_TEXT_SEARCH: QueryIntent.QUESTION_ANSWERING
    }
    
    # Generic semantic check for summarization synonyms to prevent phrase-sensitivity
    summary_synonyms = ["summarize", "summarise", "summary", "overview", "synopsis", "tldr", "brief", "main points", "key takeaways", "what was discussed", "kya baatein", "kya discuss", "details of call"]
    is_generic_summary = any(syn in question_lower for syn in summary_synonyms)

    if is_generic_summary and intent == INTENT_TEXT_SEARCH:
        target_intent_enum = QueryIntent.SUMMARIZATION
        target_scope_enum = RequestScope.SUMMARY
    else:
        target_intent_enum = intent_enum_map.get(intent, QueryIntent.QUESTION_ANSWERING)
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
            target_scope_enum = scope_enum_map.get(request_scope, RequestScope.QUESTION_ANSWER)

    source_is_explicit = bool(source_hint) or unresolved_explicit_source
    
    # Check if the resolved source actually exists in the database index or filesystem
    indexed_lower_files = [f.lower() for f in indexed_files] if indexed_files else []
    resolved_file_exists = False
    if canonical_source_id and not str(canonical_source_id).startswith("UNRESOLVED_"):
        base_name = os.path.basename(canonical_source_id).lower()
        if base_name in indexed_lower_files or os.path.exists(canonical_source_id):
            resolved_file_exists = True

    source_is_resolved = resolved_file_exists and not unresolved_explicit_source

    plan = QueryPlan(
        raw_query=question,
        normalized_query=question_lower,
        intent=target_intent_enum,
        scope=target_scope_enum,
        modality=target_modality_enum,
        source_spec=SourceSpec(
            source_hint=source_hint,
            canonical_path=canonical_source_id if resolved_file_exists else None,
            is_explicit=source_is_explicit,
            is_resolved=source_is_resolved,
            confidence=source_confidence if resolved_file_exists else 0.0
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