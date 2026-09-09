import re
import os
from app.storage.lancedb_store import get_hash_table


def get_indexed_filenames():
    """
    Helper function to get all indexed filenames from LanceDB processed_files table.
    Returns a list of filename strings.
    """
    hash_table = get_hash_table()
    if hash_table is None:
        return []

    df = hash_table.to_pandas()
    if df.empty:
        return []

    filenames = []
    for path_str in df["path"].dropna().tolist():
        base_name = os.path.basename(path_str)
        if base_name and base_name not in filenames:
            filenames.append(base_name)

    return filenames


def find_best_matching_source(question_lower, indexed_files, query_modality="all", face_intent="none"):
    """
    Helper function to find the best matching indexed filename for a user query.
    Enforces robust entity matching while ignoring generic category terms like 'audio', 'video', 'resume'.
    Restricts candidate files based on query_modality and face_intent.
    """
    if not indexed_files:
        return None

    # Step 1: Define generic category terms and filler words to ignore during stem matching
    generic_media_terms = {
        "audio", "video", "image", "picture", "photo", "recording",
        "sound", "file", "document", "documents", "pdf", "docx", "xlsx", "mp3",
        "m4a", "wav", "mpeg", "txt", "excel", "spreadsheet", "resume", "cv",
        "note", "notes", "paper", "paperwork"
    }

    stop_words = {
        "the", "a", "an", "of", "in", "from", "to", "and", "or", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "for", "with",
        "on", "at", "by", "this", "that", "my", "please", "what", "who", "how", "which",
        "where", "when", "summarize", "summarise", "summary", "chapter", "section", "part",
        "file", "show", "get", "give", "said", "discussed", "listen", "recording", "content",
        "me", "can", "you", "listed", "mention", "mentioned"
    }

    # Step 2: Filter candidate files by modality if appropriate
    candidate_files = indexed_files
    image_exts = (".jpg", ".jpeg", ".png", ".webp")
    audio_exts = (".m4a", ".mp3", ".wav", ".mpeg", ".aac", ".flac")

    if query_modality == "image" or face_intent in ["face_search", "face_identification"]:
        candidate_files = [f for f in indexed_files if f.lower().endswith(image_exts)]
    elif query_modality == "audio":
        candidate_files = [f for f in indexed_files if f.lower().endswith(audio_exts)]

    # Step 3: Check for exact full filename in query (e.g. "Accounts.m4a", "audio.mpeg")
    for filename in candidate_files:
        filename_lower = filename.lower()
        if filename_lower in question_lower:
            return filename

    # Step 4: Extract meaningful non-generic query tokens
    query_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", question_lower)
    meaningful_tokens = []
    for token in query_tokens:
        token_lower = token.lower()
        if len(token_lower) >= 3 and token_lower not in stop_words and token_lower not in generic_media_terms:
            meaningful_tokens.append(token_lower)

    # Step 5: Match meaningful query tokens against candidate filename tokens
    best_match = None
    best_score = 0

    for filename in candidate_files:
        filename_lower = filename.lower()
        filename_stem = os.path.splitext(filename_lower)[0]
        filename_words = set(re.split(r"[\s_\-\.]+", filename_lower))
        stem_words = set(re.split(r"[\s_\-\.]+", filename_stem))

        score = 0
        for token in meaningful_tokens:
            if token in filename_words or token in stem_words:
                score = score + 10

        if score > best_score:
            best_score = score
            best_match = filename

    if best_match is not None and best_score > 0:
        return best_match

    # Step 6: Check exact stem matches for non-generic stems
    for filename in candidate_files:
        filename_lower = filename.lower()
        filename_stem = os.path.splitext(filename_lower)[0]

        if len(filename_stem) > 3 and filename_stem not in generic_media_terms and filename_stem not in stop_words:
            if filename_stem in question_lower:
                return filename

    return None


def resolve_canonical_source_id(source_hint):
    """
    Resolves source_hint string to the canonical source_id / full filepath in LanceDB index.
    Returns canonical_source_id string or None.
    """
    if not source_hint or source_hint == "UNRESOLVED_AUDIO_SOURCE":
        return None

    hash_table = get_hash_table()
    if hash_table is not None and hash_table.count_rows() > 0:
        df = hash_table.to_pandas()
        if not df.empty and "path" in df.columns:
            for path_str in df["path"].dropna().tolist():
                if os.path.basename(path_str).lower() == source_hint.lower():
                    return path_str
                if source_hint.lower() in path_str.lower():
                    return path_str

    # Fallback search in watched_folder or current workspace
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    watched_dir = os.path.join(base_dir, "watched_folder")
    if os.path.exists(watched_dir):
        for fn in os.listdir(watched_dir):
            if fn.lower() == source_hint.lower():
                return os.path.join(watched_dir, fn)

    for fn in os.listdir(base_dir):
        if fn.lower() == source_hint.lower():
            return os.path.join(base_dir, fn)

    return source_hint


def analyze_query(question, indexed_files=None):
    """
    Analyzes the user query to detect:
    - Intent: 'correction', 'face_intent', 'summarization', 'speaker_analysis', 'question_answering'
    - Face Intent: 'face_identification', 'face_registration', 'face_search', 'none'
    - Modality: 'audio', 'image', 'video', 'pdf', 'docx', 'spreadsheet', 'text', 'all'
    - Source Hint: Exact filename if mentioned (e.g. 'Accounts.m4a', 'mummy.jpg')
    - Canonical Source ID: Authoritative full path or canonical identifier
    - Visual QA: True if query asks visual questions about an image
    """
    question_lower = question.strip().lower()

    # Step 1: Fetch indexed filenames if not passed explicitly
    if indexed_files is None:
        indexed_files = get_indexed_filenames()

    # Step 2: Check for Correction Intent
    correction_patterns = [
        r"\byou\s+are\s+incorrect\b",
        r"\byou\s+are\s+wrong\b",
        r"\bwrong\s+answer\b",
        r"\bwrong\s+file\b",
        r"\bwrong\s+source\b",
        r"\bthis\s+is\s+incorrect\b",
        r"\bthat\s+is\s+not\s+correct\b",
        r"\bcorrect\s+it\b",
        r"\bplease\s+correct\b",
        r"\bthe\s+correct\s+file\s+is\b",
        r"\bthe\s+source\s+should\s+be\b",
        r"\bthat\s+answer\s+is\s+wrong\b",
        r"\bnot\s+this\s+file\b",
        r"\bnot\s+this\b",
        r"\buse\s+.*\s+instead\b",
        r"\bremember\s+this\s+correction\b",
        r"\bfile\s+source\s+of\s+.*above\s+query\b",
        r"\bprevious\s+answer\s+is\s+wrong\b",
        r"\byour\s+answer\s+is\s+wrong\b",
        r"\bbut\s+there\s+are\b",
        r"\bthere\s+are\s+two\b",
        r"\bone\s+is\s+.*\s+second\s+one\s+is\b",
        r"\bone\s+is\s+.*\s+second\s+is\b",
        r"\bthe\s+image\s+contains\b",
        r"\byou\s+missed\b",
        r"\bthe\s+correct\s+answer\s+is\b"
    ]

    is_correction = False
    for pattern in correction_patterns:
        if re.search(pattern, question_lower):
            is_correction = True
            break

    # Step 3: Extract Source Hint matching indexed files or colloquial terms
    source_hint = None

    colloquial_mappings = {
        "account audio": "Accounts.m4a",
        "accounts audio": "Accounts.m4a",
        "audio accounts": "Accounts.m4a",
        "audio account": "Accounts.m4a",
        "accounts.m4a": "Accounts.m4a",
        "account.m4a": "Accounts.m4a",
        "mlk audio": "MLKDream.mp3",
        "mlk dream": "MLKDream.mp3",
        "audio.mpeg": "audio.mpeg",
        "germany it companies": "Germany_IT_Companies_Rashmi_2026.xlsx",
        "germany_it_companies_rashmi_2026.xlsx": "Germany_IT_Companies_Rashmi_2026.xlsx",
        "mummy image": "mummy.jpg",
        "mummy picture": "mummy.jpg",
        "mummy photo": "mummy.jpg",
        "mummy.jpg": "mummy.jpg",
        "mummy": "mummy.jpg",
        "mom image": "mummy.jpg",
        "mom picture": "mummy.jpg",
        "mom photo": "mummy.jpg",
        "mom": "mummy.jpg",
        "mother image": "mummy.jpg",
        "mother picture": "mummy.jpg",
        "mother photo": "mummy.jpg",
        "mother": "mummy.jpg"
    }

    for colloquial_key, target_filename in colloquial_mappings.items():
        if re.search(r"\b" + re.escape(colloquial_key) + r"\b", question_lower):
            source_hint = target_filename
            break

    if source_hint is None:
        file_match = re.search(r"\b([a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|m4a|mp3|wav|mpeg|pdf|docx|xlsx))\b", question_lower)
        if file_match:
            source_hint = file_match.group(1)

    if source_hint is None:
        source_hint = find_best_matching_source(question_lower, indexed_files)

    # Step 3b: If query asks for an audio summary of an unknown/unmatched entity, mark as unresolved
    is_audio_term = any(term in question_lower for term in ["audio", "recording", "sound", "voice", "speaker", "m4a", "mp3", "wav"])
    is_summary_term = any(term in question_lower for term in ["summarize", "summarise", "summary", "overview", "what was said", "what is discussed"])

    if source_hint is None and is_audio_term and is_summary_term:
        query_words = re.findall(r"\b[a-zA-Z0-9_]+\b", question_lower)
        generic_terms = {
            "audio", "sound", "recording", "speaker", "voice", "m4a", "mp3", "wav",
            "summarize", "summarise", "summary", "overview", "the", "a", "an", "of", "in", "what", "is", "was"
        }
        entity_words = [w for w in query_words if len(w) >= 3 and w.lower() not in generic_terms]
        if len(entity_words) > 0:
            source_hint = "UNRESOLVED_AUDIO_SOURCE"

    canonical_source_id = resolve_canonical_source_id(source_hint)

    # Step 4: Detect Modality
    modality = "all"

    audio_keywords = [
        "audio", "sound", "m4a", "mp3", "wav", "mpeg", "recording",
        "speaker", "speak", "voice", "what was said", "what did they say",
        "conversation", "listen", "girl speak", "boy speak"
    ]

    image_keywords = [
        "image", "picture", "photo", "visual", "diagram", "chart",
        "figure", "jpg", "jpeg", "png"
    ]

    video_keywords = [
        "video", "mp4", "mkv", "avi", "mov", "frame"
    ]

    pdf_keywords = ["pdf"]
    docx_keywords = ["docx", "doc", "word document"]
    spreadsheet_keywords = ["excel", "xlsx", "xls", "spreadsheet", "csv"]

    if source_hint:
        ext = os.path.splitext(source_hint)[1].lower()
        if ext in [".m4a", ".mp3", ".wav", ".mpeg", ".aac", ".flac"]:
            modality = "audio"
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            modality = "image"
        elif ext in [".mp4", ".mkv", ".avi", ".mov"]:
            modality = "video"
        elif ext in [".pdf"]:
            modality = "pdf"
        elif ext in [".docx", ".doc"]:
            modality = "docx"
        elif ext in [".xlsx", ".xls"]:
            modality = "spreadsheet"

    if modality == "all":
        if any(w in question_lower for w in audio_keywords):
            modality = "audio"
        elif any(w in question_lower for w in image_keywords):
            modality = "image"
        elif any(w in question_lower for w in video_keywords):
            modality = "video"
        elif any(w in question_lower for w in pdf_keywords):
            modality = "pdf"
        elif any(w in question_lower for w in docx_keywords):
            modality = "docx"
        elif any(w in question_lower for w in spreadsheet_keywords):
            modality = "spreadsheet"

    # Step 5: Detect Face Intent & Visual QA
    face_intent = "none"

    face_id_patterns = [
        r"\bwho\s+is\s+this\b", r"\bwho\s+is\s+the\b", r"\bwho\s+is\s+in\b",
        r"\bdo\s+you\s+recognize\b", r"\bis\s+this\s+my\b", r"\bis\s+.*\s+in\s+this\s+image\b",
        r"\bwho\s+are\s+the\s+people\b", r"\bwho\s+is\s+standing\b"
    ]

    face_search_patterns = [
        r"\bfind\s+(all\s+)?(images|pictures|photos)\s+(of|containing|with)\b",
        r"\bshow\s+(me\s+)?(images|pictures|photos)\s+(of|containing|with)\b",
        r"\bpicture\s+of\b", r"\bphoto\s+of\b", r"\bimage\s+of\b"
    ]

    face_registration_patterns = [
        r"^this\s+is\s+my\s+[a-z0-9_]+$",
        r"^this\s+is\s+[a-z0-9_]+$",
        r"^her\s+name\s+is\s+[a-z0-9_]+$",
        r"^his\s+name\s+is\s+[a-z0-9_]+$"
    ]

    if any(re.search(pat, question_lower) for pat in face_registration_patterns):
        face_intent = "face_registration"
    elif any(re.search(pat, question_lower) for pat in face_id_patterns):
        face_intent = "face_identification"
    elif any(re.search(pat, question_lower) for pat in face_search_patterns):
        face_intent = "face_search"

    is_visual_qa = False
    vqa_words = [
        "how many people", "what color", "what is on the left", "what is on the right",
        "clothing", "wearing", "describe the image", "visible in"
    ]
    if modality == "image" and any(vw in question_lower for vw in vqa_words):
        is_visual_qa = True

    # Step 6: Detect Speaker Reference
    speaker_reference = None
    speaker_match = re.search(r"\bspeaker\s*(\d+|[a-zA-Z])\b", question_lower)
    if speaker_match:
        speaker_reference = f"Speaker {speaker_match.group(1).upper()}"

    # Step 7: Determine Overall Query Intent
    if is_correction:
        intent = "correction"
    elif face_intent != "none":
        intent = "face_intent"
    elif any(re.search(r"\b" + p + r"\b", question_lower) for p in ["summarize", "summarise", "summary", "overview"]):
        intent = "summarization"
    elif speaker_reference or "who speaks" in question_lower or "girl speak" in question_lower or "boy speak" in question_lower:
        intent = "speaker_analysis"
    else:
        intent = "question_answering"

    # Step 8: Build Correction Details if Intent is Correction
    correction_details = {}
    if is_correction:
        correction_details = {
            "is_correction": True,
            "correct_source": source_hint,
            "corrected_answer": ""
        }
        answer_match = re.search(r"correct answer is\s+(.*)", question, re.IGNORECASE)
        if answer_match:
            correction_details["corrected_answer"] = answer_match.group(1).strip()

    analysis_result = {
        "intent": intent,
        "face_intent": face_intent,
        "is_visual_qa": is_visual_qa,
        "modality": modality,
        "source_hint": source_hint,
        "canonical_source_id": canonical_source_id,
        "speaker_reference": speaker_reference,
        "is_correction": is_correction,
        "correction_details": correction_details,
        "needs_database": True
    }

    return analysis_result