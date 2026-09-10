import re
import os
from app.storage.lancedb_store import get_hash_table

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
    "audio", "image", "images", "video", "picture", "photo", "recording",
    "sound", "file", "files", "document", "documents", "pdf", "docx", "xlsx", "mp3",
    "m4a", "wav", "mpeg", "txt", "excel", "list", "name", "names", "page", "pages"
}

STOP_WORDS_SET = {
    "the", "a", "an", "of", "in", "from", "to", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "for", "with",
    "on", "at", "by", "this", "that", "my", "please", "what", "who", "how", "which",
    "where", "when", "summarize", "summarise", "summary", "chapter", "section", "part",
    "show", "get", "give", "said", "discussed", "listen", "recording", "content", "me",
    "can", "you", "listed", "mention", "mentioned", "many", "much", "count", "number",
    "there", "tell", "read"
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
    Normalizes text by lowercasing and replacing punctuation/symbols with single spaces.
    Example: 'Behari_lal-call.m4a' -> 'behari lal call m4a'
    """
    if not text_input:
        return ""
    cleaned_text = re.sub(r"[_\-\.\:\,\!\?\/]+", " ", str(text_input).lower())
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
    Extracts non-stopword entity tokens from a text input.
    """
    normalized = normalize_string(text_input)
    raw_tokens = normalized.split()
    filtered_tokens = []
    for token in raw_tokens:
        if len(token) >= 2 and token not in STOP_WORDS_SET and token not in GENERIC_MEDIA_TERMS:
            filtered_tokens.append(token)

    return filtered_tokens


def find_best_matching_source(question_lower, indexed_files, query_modality="all", face_intent="none"):
    """
    Data-driven filename and source resolution pipeline.
    Preserves exact original filename casing from indexed_files.
    
    Resolution Strategy:
    1. Exact full filename in query (e.g. Accounts.m4a, audio.mpeg).
    2. Entity token matching (e.g. 'jethalal' -> 'Jethalal ने...mp3', 'behari lal' -> 'Behari_lal_call.m4a').
    3. Exact normalized stem match (handling digit variants like 'dense 2' -> 'dense2.webp', excluding generic media terms).
    4. Compact stem substring fallback.
    """
    if not indexed_files:
        return None

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

    # Stage 1: Exact Full Filename in Query (case-insensitive substring match)
    for candidate in candidate_files:
        if candidate.lower() in question_lower:
            return candidate

    # Stage 2: Entity Token Matching (Highest priority for specific entity names)
    question_entity_tokens = extract_entity_tokens(question_lower)

    if question_entity_tokens:
        scored_candidates = []
        for candidate in candidate_files:
            file_stem = os.path.splitext(candidate)[0]
            stem_tokens = set(extract_entity_tokens(file_stem))

            exact_match_count = 0
            partial_match_count = 0
            for q_token in question_entity_tokens:
                for s_token in stem_tokens:
                    if q_token == s_token:
                        exact_match_count += 1
                        break
                    elif (len(q_token) >= 4 and q_token in s_token) or (len(s_token) >= 4 and s_token in q_token):
                        partial_match_count += 1
                        break

            if exact_match_count > 0 or partial_match_count > 0:
                scored_candidates.append((exact_match_count, partial_match_count, -len(stem_tokens), candidate))

        if scored_candidates:
            scored_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            return scored_candidates[0][3]

    # Stage 3: Exact Stem & Compact Stem Matching (excluding generic media terms)
    best_exact_match = None
    best_exact_length = 0

    for candidate in candidate_files:
        file_stem = os.path.splitext(candidate)[0]
        normalized_stem = normalize_string(file_stem)
        compact_stem = compact_alphanumeric(file_stem)

        # Skip generic media terms like 'audio', 'image', 'file' unless explicitly matched in Stage 1
        if normalized_stem in GENERIC_MEDIA_TERMS:
            continue

        word_boundary_pattern = r"\b" + re.escape(normalized_stem) + r"\b"
        if re.search(word_boundary_pattern, normalized_question):
            if len(normalized_stem) > best_exact_length:
                best_exact_match = candidate
                best_exact_length = len(normalized_stem)

        if not best_exact_match and compact_stem and len(compact_stem) >= 3:
            if compact_stem == compact_question or re.search(r"\b" + re.escape(normalized_stem) + r"\b", normalized_question):
                return candidate

            digit_match = re.search(r"(\d+)$", file_stem)
            if digit_match:
                digit_number = digit_match.group(1)
                base_part = file_stem[:len(file_stem) - len(digit_number)]
                norm_base = normalize_string(base_part)
                digit_pattern = r"\b" + re.escape(norm_base) + r"\s*" + re.escape(digit_number) + r"\b"
                if re.search(digit_pattern, normalized_question):
                    return candidate

    if best_exact_match:
        return best_exact_match

    # Stage 4: Substring Fallback for compact stems (non-generic)
    for candidate in candidate_files:
        file_stem = os.path.splitext(candidate)[0]
        compact_stem = compact_alphanumeric(file_stem)
        if compact_stem and len(compact_stem) >= 4 and compact_stem not in GENERIC_MEDIA_TERMS and compact_stem in compact_question:
            return candidate

    return None


def resolve_canonical_source_id(source_hint):
    """
    Resolves a source_hint filename string to the canonical full filepath in LanceDB index or filesystem.
    """
    if not source_hint or source_hint == "UNRESOLVED_AUDIO_SOURCE":
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


def analyze_query(question, indexed_files=None):
    """
    Analyzes the user query and produces structured classification output containing:
    - intent: explicit Intent Taxonomy constant
    - temporal_intent: temporal category
    - face_intent: face processing category
    - modality: audio | image | video | pdf | docx | spreadsheet | text | all
    - source_hint: resolved filename string if query specifies a source
    - canonical_source_id: full resolved path to source file
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

    # Step 2: Extract Temporal Intent
    temporal_intent = "none"
    recent_patterns = [r"\badded\s+recently\b", r"\brecently\s+added\b", r"\blatest\s+files\b", r"\brecent\s+files\b", r"\brecent\s+images\b"]
    datewise_patterns = [r"\bdate-wise\s+list\b", r"\bdatewise\s+list\b", r"\bby\s+date\b"]
    date_filter_patterns = [r"\badded\s+today\b", r"\badded\s+yesterday\b", r"\bwere\s+added\s+today\b", r"\bwere\s+added\s+yesterday\b"]
    count_temporal_patterns = [r"\bhow\s+many\s+files\s+added\b", r"\bhow\s+many\s+images\s+added\b", r"\bhow\s+many\s+files\s+were\s+added\b"]

    if any(re.search(pat, question_lower) for pat in datewise_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY
    elif any(re.search(pat, question_lower) for pat in count_temporal_patterns):
        temporal_intent = INTENT_FILE_COUNT
    elif any(re.search(pat, question_lower) for pat in date_filter_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY
    elif any(re.search(pat, question_lower) for pat in recent_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY

    # Step 3: Modality Detection
    modality = "all"
    audio_keywords = ["audio", "sound", "m4a", "mp3", "wav", "mpeg", "recording", "speaker", "speak", "voice", "con call", "talk"]
    image_keywords = ["image", "images", "picture", "pictures", "photo", "photos", "visual", "diagram", "chart", "figure", "jpg", "jpeg", "png", "webp"]

    if any(kw in question_lower for kw in audio_keywords):
        modality = "audio"
    elif any(kw in question_lower for kw in image_keywords):
        modality = "image"

    # Step 4: Source Hint Resolution
    source_hint = None
    if temporal_intent == "none":
        file_match = re.search(r"\b([a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|m4a|mp3|wav|mpeg|pdf|docx|xlsx))\b", question_lower)
        if file_match:
            matched_str = file_match.group(1)
            for fname in indexed_files:
                if fname.lower() == matched_str.lower():
                    source_hint = fname
                    break
            if source_hint is None:
                source_hint = matched_str

        if source_hint is None:
            source_hint = find_best_matching_source(question_lower, indexed_files, query_modality=modality)

    # Check for Unresolved Audio Source
    is_audio_term = any(term in question_lower for term in ["audio", "recording", "sound", "voice", "m4a", "mp3", "wav"])
    is_summary_term = any(term in question_lower for term in ["summarize", "summarise", "summary", "overview"])

    if source_hint is None and is_audio_term and is_summary_term and temporal_intent == "none":
        entity_words = extract_entity_tokens(question_lower)
        if len(entity_words) > 0:
            source_hint = "UNRESOLVED_AUDIO_SOURCE"

    # Update modality if source_hint has explicit file extension
    if source_hint and source_hint != "UNRESOLVED_AUDIO_SOURCE":
        extension = os.path.splitext(source_hint)[1].lower()
        if extension in [".m4a", ".mp3", ".wav", ".mpeg"]:
            modality = "audio"
        elif extension in [".jpg", ".jpeg", ".png", ".webp"]:
            modality = "image"

    canonical_source_id = resolve_canonical_source_id(source_hint)

    # Step 5: Detect Intent Category
    image_name_query_patterns = [
        r"\bnames\s+of\s+.*images?\b",
        r"\blist\s+the\s+name\s+images?\b",
        r"\bimages?\s+which\s+have\s+named\s+by\b",
        r"\blist\s+images?\s+named\b",
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
        r"\bwho\s+is\s+speaking\b"
    ]
    is_speaker_query = any(re.search(pat, question_lower) for pat in speaker_patterns)

    ocr_phrases = [
        "text in the image", "text in image", "what is the text", "read text",
        "ocr text", "words in the image", "writing in the image", "text of image",
        "all the names in", "all names in", "all the text in", "all text in",
        "text from dense", "text from dense2", "text from dense3", "content of dense image"
    ]
    is_ocr_query = any(op in question_lower for op in ocr_phrases)

    image_summary_phrases = [
        "summarize the image", "summarise the image", "summary of the image",
        "summarize image", "summarise image", "describe the image", "describe image",
        "summarize dense", "summarize dense2", "summarize dense3"
    ]
    is_image_summary_query = any(isp in question_lower for isp in image_summary_phrases)

    is_full_transcript = any(p in question_lower for p in ["full transcript", "entire transcript", "complete transcript", "give me the transcript"])

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
    elif is_full_transcript:
        intent = INTENT_AUDIO_TRANSCRIPT
    elif is_ocr_query:
        intent = INTENT_IMAGE_OCR
    elif is_image_summary_query:
        intent = INTENT_IMAGE_SUMMARY
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
        "temporal_intent": temporal_intent,
        "face_intent": "none",
        "is_visual_qa": is_image_summary_query or is_ocr_query,
        "is_ocr_query": is_ocr_query,
        "is_image_summary_query": is_image_summary_query,
        "modality": modality,
        "source_hint": source_hint,
        "canonical_source_id": canonical_source_id,
        "is_correction": is_correction,
        "needs_database": True
    }

    return analysis_result


def resolve_canonical_source_id(source_hint):
    """
    Resolves a source_hint filename string to the canonical full filepath in LanceDB index or filesystem.
    """
    if not source_hint or source_hint == "UNRESOLVED_AUDIO_SOURCE":
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


def analyze_query(question, indexed_files=None):
    """
    Analyzes the user query and produces structured classification output containing:
    - intent: explicit Intent Taxonomy constant
    - temporal_intent: temporal category
    - face_intent: face processing category
    - modality: audio | image | video | pdf | docx | spreadsheet | text | all
    - source_hint: resolved filename string if query specifies a source
    - canonical_source_id: full resolved path to source file
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

    # Step 2: Extract Temporal Intent
    temporal_intent = "none"
    recent_patterns = [r"\badded\s+recently\b", r"\brecently\s+added\b", r"\blatest\s+files\b", r"\brecent\s+files\b", r"\brecent\s+images\b"]
    datewise_patterns = [r"\bdate-wise\s+list\b", r"\bdatewise\s+list\b", r"\bby\s+date\b"]
    date_filter_patterns = [r"\badded\s+today\b", r"\badded\s+yesterday\b", r"\bwere\s+added\s+today\b", r"\bwere\s+added\s+yesterday\b"]
    count_temporal_patterns = [r"\bhow\s+many\s+files\s+added\b", r"\bhow\s+many\s+images\s+added\b", r"\bhow\s+many\s+files\s+were\s+added\b"]

    if any(re.search(pat, question_lower) for pat in datewise_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY
    elif any(re.search(pat, question_lower) for pat in count_temporal_patterns):
        temporal_intent = INTENT_FILE_COUNT
    elif any(re.search(pat, question_lower) for pat in date_filter_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY
    elif any(re.search(pat, question_lower) for pat in recent_patterns):
        temporal_intent = INTENT_TEMPORAL_FILE_QUERY

    # Step 3: Modality Detection
    modality = "all"
    audio_keywords = ["audio", "sound", "m4a", "mp3", "wav", "mpeg", "recording", "speaker", "speak", "voice", "con call", "talk"]
    image_keywords = ["image", "images", "picture", "pictures", "photo", "photos", "visual", "diagram", "chart", "figure", "jpg", "jpeg", "png", "webp"]

    if any(kw in question_lower for kw in audio_keywords):
        modality = "audio"
    elif any(kw in question_lower for kw in image_keywords):
        modality = "image"

    # Step 4: Source Hint Resolution
    source_hint = None
    if temporal_intent == "none":
        file_match = re.search(r"\b([a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|m4a|mp3|wav|mpeg|pdf|docx|xlsx))\b", question_lower)
        if file_match:
            source_hint = file_match.group(1)

        if source_hint is None:
            source_hint = find_best_matching_source(question_lower, indexed_files, query_modality=modality)

    # Check for Unresolved Audio Source
    is_audio_term = any(term in question_lower for term in ["audio", "recording", "sound", "voice", "m4a", "mp3", "wav"])
    is_summary_term = any(term in question_lower for term in ["summarize", "summarise", "summary", "overview"])

    if source_hint is None and is_audio_term and is_summary_term and temporal_intent == "none":
        entity_words = extract_entity_tokens(question_lower)
        if len(entity_words) > 0:
            source_hint = "UNRESOLVED_AUDIO_SOURCE"

    # Update modality if source_hint has explicit file extension
    if source_hint and source_hint != "UNRESOLVED_AUDIO_SOURCE":
        extension = os.path.splitext(source_hint)[1].lower()
        if extension in [".m4a", ".mp3", ".wav", ".mpeg"]:
            modality = "audio"
        elif extension in [".jpg", ".jpeg", ".png", ".webp"]:
            modality = "image"

    canonical_source_id = resolve_canonical_source_id(source_hint)

    # Step 5: Detect Intent Category
    # Check Image Filename Metadata Query Intent (e.g. 'give me the names of dense images')
    image_name_query_patterns = [
        r"\bnames\s+of\s+.*images?\b",
        r"\blist\s+the\s+name\s+images?\b",
        r"\bimages?\s+which\s+have\s+named\s+by\b",
        r"\blist\s+images?\s+named\b",
        r"\bfilenames?\s+of\b"
    ]
    is_image_filename_query = any(re.search(pat, question_lower) for pat in image_name_query_patterns)

    # Check Image Count Query Intent (e.g. 'how many images I have by the names of dense?')
    image_count_query_patterns = [
        r"\bhow\s+many\s+images?\s+.*named\b",
        r"\bhow\s+many\s+images?\s+.*by\s+the\s+names?\b",
        r"\bcount\s+of\s+images?\s+named\b"
    ]
    is_image_count_query = any(re.search(pat, question_lower) for pat in image_count_query_patterns)

    # Check Chunk / File Metadata Query Intent (e.g. 'how many chunks does behari lal have?')
    chunk_metadata_patterns = [
        r"\bhow\s+many\s+chunks\b",
        r"\bchunk\s+count\b",
        r"\bwhen\s+was\s+this\s+file\s+added\b"
    ]
    is_chunk_metadata_query = any(re.search(pat, question_lower) for pat in chunk_metadata_patterns)

    # Check Speaker Query Intent (e.g. 'who is speakers in behari lal file?')
    speaker_patterns = [
        r"\bwho\s+is\s+speakers?\b",
        r"\bwho\s+are\s+the\s+speakers?\b",
        r"\bhow\s+many\s+persons?\s+are\s+talking\b",
        r"\bhow\s+many\s+people\s+are\s+talking\b",
        r"\bis\s+there\s+any\s+lady\s+talk\b",
        r"\bis\s+a\s+woman\s+speaking\b",
        r"\bwho\s+is\s+speaking\b"
    ]
    is_speaker_query = any(re.search(pat, question_lower) for pat in speaker_patterns)

    # Check OCR & Text Extraction Intent (e.g. 'give me all the content of dense image')
    ocr_phrases = [
        "text in the image", "text in image", "what is the text", "read text",
        "ocr text", "words in the image", "writing in the image", "text of image",
        "all the names in", "all names in", "all the text in", "all text in",
        "text from dense", "text from dense2", "text from dense3", "content of dense image"
    ]
    is_ocr_query = any(op in question_lower for op in ocr_phrases)

    # Check Image Summary / Visual Query Intent
    image_summary_phrases = [
        "summarize the image", "summarise the image", "summary of the image",
        "summarize image", "summarise image", "describe the image", "describe image",
        "summarize dense", "summarize dense2", "summarize dense3"
    ]
    is_image_summary_query = any(isp in question_lower for isp in image_summary_phrases)

    # Check Full Transcript Intent
    is_full_transcript = any(p in question_lower for p in ["full transcript", "entire transcript", "complete transcript", "give me the transcript"])

    # Determine Final Intent
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
    elif is_full_transcript:
        intent = INTENT_AUDIO_TRANSCRIPT
    elif is_ocr_query:
        intent = INTENT_IMAGE_OCR
    elif is_image_summary_query:
        intent = INTENT_IMAGE_SUMMARY
    elif any(word in question_lower for word in ["summarize", "summarise", "summary", "overview"]):
        if modality == "audio":
            intent = INTENT_AUDIO_SUMMARY
        elif modality == "image":
            intent = INTENT_IMAGE_SUMMARY
        else:
            intent = INTENT_DOCUMENT_SUMMARY
    else:
        intent = INTENT_TEXT_SEARCH

    # Build Analysis Output Dictionary
    analysis_result = {
        "intent": intent,
        "temporal_intent": temporal_intent,
        "face_intent": "none",
        "is_visual_qa": is_image_summary_query or is_ocr_query,
        "is_ocr_query": is_ocr_query,
        "is_image_summary_query": is_image_summary_query,
        "modality": modality,
        "source_hint": source_hint,
        "canonical_source_id": canonical_source_id,
        "is_correction": is_correction,
        "needs_database": True
    }

    return analysis_result