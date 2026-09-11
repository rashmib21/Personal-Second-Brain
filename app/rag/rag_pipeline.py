import os
import re
from app.search.vector_search import search, get_full_transcript_for_source, get_stored_ocr_text_for_image
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.storage.lancedb_store import (
    get_hash_table,
    get_image_table,
    store_feedback,
    search_feedback
)
from app.query.query_analyzer import analyze_query
from app.llm.summary_client import summarize_text
from app.search.summary_search import search_for_summary, search_for_book_summary
from app.rag.image_summarizer import summarize_image
from app.search.object_search import search_images_by_object
from app.search.face_search import is_face_search_query, search_images_by_face, extract_person_name_from_question
from app.services.interaction_state import get_last_interaction, update_last_interaction
from config import DEBUG



def validate_content_grounding(context_text, answer_text, question="", source_path=""):
    """
    Redesigned Content Grounding Validation.
    Evaluates:
    1. Intent fulfillment / Non-refusal when context exists
    2. Fact/Claim support without false flags for query/source words
    3. Contradiction / Hallucination detection
    Returns tuple: (is_grounded: bool, unsupported_claims: list)
    """
    if not answer_text:
        # print("\n===== CONTENT GROUNDING VALIDATION =====")
        # print("Content grounding: FAIL (Empty answer)")
        return False, ["empty_answer"]

    ans_lower = answer_text.lower()
    ctx_lower = (context_text or "").lower()
    q_lower = (question or "").lower()

    # 1. Refusal check: If context text exists and has >50 chars, but answer refuses
    refusal_phrases = [
        "couldn't find any information", "could not find any information",
        "don't have enough information", "do not have enough information",
        "cannot provide", "can't provide", "no information was found",
        "please upload", "not provided a specific image"
    ]
    if context_text and len(context_text.strip()) > 50:
        if any(rp in ans_lower for rp in refusal_phrases):
            # print("\n===== CONTENT GROUNDING VALIDATION =====")
            # print("Content grounding: FAIL (False refusal when relevant context exists)")
            return False, ["false_refusal"]

    # 2. Extract capitalized proper nouns from answer for claim checking
    answer_words = re.findall(r"\b[A-Z][a-z]{2,}\b", answer_text)

    # Tokens from query and resolved source filename must NOT be treated as unsupported
    query_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", q_lower))
    source_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", os.path.basename(source_path).lower())) if source_path else set()

    ignored_words = {
        "The", "This", "Here", "There", "Yes", "No", "Please", "Answer", "File",
        "Source", "Retrieved", "Note", "A", "An", "In", "On", "At", "For", "With", "By",
        "Is", "Are", "Was", "Were", "It", "Its", "From", "About", "Has", "Have", "Had",
        "Call", "Audio", "Image", "Text", "Document", "Summary", "Transcript"
    }

    unsupported = []
    for word in set(answer_words):
        w_low = word.lower()
        if word in ignored_words or w_low in query_tokens or w_low in source_tokens:
            continue
        if w_low not in ctx_lower:
            unsupported.append(word)

    # print("\n===== CONTENT GROUNDING VALIDATION =====")
    # if unsupported:
    #     print(f"Content grounding: WARN (Unsupported terms: {unsupported})")
    # else:
    #     print("Content grounding: PASS")

    return (len(unsupported) == 0), unsupported


def handle_temporal_query(question, analysis):
    """
    Handles file listing, temporal file queries, and file count queries deterministically
    by querying the LanceDB processed_files metadata table directly without LLM invocation.
    Supports dynamic date parsing, modality filtering, limit extraction, and deduplication.
    """
    from datetime import datetime
    from app.query.query_analyzer import INTENT_FILE_COUNT, INTENT_FILE_LIST, INTENT_TEMPORAL_FILE_QUERY

    # Step 1: Retrieve hash table from LanceDB database
    hash_table = get_hash_table()
    if hash_table is None or hash_table.count_rows() == 0:
        return "No files have been added to the Second Brain database.", [], 0

    database_dataframe = hash_table.to_pandas()
    if database_dataframe.empty or "path" not in database_dataframe.columns:
        return "No files have been added to the Second Brain database.", [], 0

    # Step 2: Extract classification metadata from analysis dictionary
    modality = analysis.get("modality", "all")
    intent = analysis.get("intent", "TEMPORAL_FILE_QUERY")
    temporal_intent = analysis.get("temporal_intent", "none")
    start_datetime = analysis.get("start_datetime")
    end_datetime = analysis.get("end_datetime")
    date_label = analysis.get("date_label")
    extracted_limit = analysis.get("extracted_limit")

    # Step 3: Define modality file extension filters
    audio_extensions = (".m4a", ".mp3", ".wav", ".mpeg", ".aac", ".flac", ".ogg")
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    pdf_extensions = (".pdf",)
    docx_extensions = (".docx", ".doc")
    video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".webm")
    text_extensions = (".txt", ".md", ".csv")

    # Step 4: Group and deduplicate database records by file base name
    # Re-ingestion can produce multiple entries; retain the record with the latest timestamp.
    unique_file_map = {}

    for row_index, dataframe_row in database_dataframe.iterrows():
        path_string = str(dataframe_row.get("path", "")).strip()
        if not path_string:
            continue

        base_filename = os.path.basename(path_string)

        # Retrieve created_at timestamp string
        created_at_raw = str(dataframe_row.get("created_at", "")).strip()

        # Parse timestamp into a datetime object for comparison and filtering
        parsed_datetime = None
        if created_at_raw:
            try:
                clean_timestamp_string = created_at_raw.replace("T", " ").split(".")[0]
                parsed_datetime = datetime.strptime(clean_timestamp_string, "%Y-%m-%d %H:%M:%S")
            except Exception:
                parsed_datetime = None

        # Fallback to filesystem modification time if database timestamp is missing or unparseable
        if parsed_datetime is None:
            if os.path.exists(path_string):
                try:
                    file_mtime = os.path.getmtime(path_string)
                    parsed_datetime = datetime.fromtimestamp(file_mtime)
                except Exception:
                    parsed_datetime = None

        if parsed_datetime is None:
            parsed_datetime = datetime.now()

        # Deduplicate by retaining the record with the newest timestamp
        if base_filename in unique_file_map:
            existing_record = unique_file_map[base_filename]
            if parsed_datetime > existing_record["datetime"]:
                unique_file_map[base_filename] = {
                    "filename": base_filename,
                    "path": path_string,
                    "datetime": parsed_datetime,
                    "created_at_raw": created_at_raw
                }
        else:
            unique_file_map[base_filename] = {
                "filename": base_filename,
                "path": path_string,
                "datetime": parsed_datetime,
                "created_at_raw": created_at_raw
            }

    file_records_list = list(unique_file_map.values())

    # Step 5: Filter records by requested modality
    filtered_records_list = []
    for file_record in file_records_list:
        filename_lower = file_record["filename"].lower()

        if modality == "audio":
            if filename_lower.endswith(audio_extensions):
                filtered_records_list.append(file_record)
        elif modality == "image":
            if filename_lower.endswith(image_extensions):
                filtered_records_list.append(file_record)
        elif modality == "pdf":
            if filename_lower.endswith(pdf_extensions):
                filtered_records_list.append(file_record)
        elif modality == "docx":
            if filename_lower.endswith(docx_extensions):
                filtered_records_list.append(file_record)
        elif modality == "video":
            if filename_lower.endswith(video_extensions):
                filtered_records_list.append(file_record)
        elif modality in ["text", "document"]:
            if filename_lower.endswith(pdf_extensions + docx_extensions + text_extensions):
                filtered_records_list.append(file_record)
        else:
            filtered_records_list.append(file_record)

    # Step 6: Filter records by target date range if specified
    if start_datetime is not None and end_datetime is not None:
        date_filtered_records = []
        for file_record in filtered_records_list:
            record_dt = file_record["datetime"]
            if start_datetime <= record_dt <= end_datetime:
                date_filtered_records.append(file_record)
        filtered_records_list = date_filtered_records

    # Step 7: Sort matching records by timestamp descending (newest first)
    filtered_records_list.sort(key=lambda item: item["datetime"], reverse=True)

    # Step 8: Handle Zero Results truthful response (No semantic search fallback!)
    if len(filtered_records_list) == 0:
        modality_label = modality if modality != "all" else "file"
        if date_label:
            message_text = f"No {modality_label} files were found for {date_label}."
        else:
            message_text = f"No {modality_label} files were found in the database."
        return message_text, [], 0

    # Step 9: Handle File Count query intent
    question_lower = question.lower()
    if intent == INTENT_FILE_COUNT or temporal_intent == INTENT_FILE_COUNT or "how many" in question_lower or "count of" in question_lower:
        total_count = len(filtered_records_list)
        modality_label = modality if modality != "all" else "file"
        if date_label:
            message_text = f"Total {modality_label} files added {date_label}: {total_count}."
        else:
            message_text = f"Total {modality_label} files added: {total_count}."
        source_filenames = [record["filename"] for record in filtered_records_list]
        return message_text, source_filenames, total_count

    # Step 10: Apply limit if specified (e.g., top N or latest 5)
    if extracted_limit is not None and extracted_limit > 0:
        filtered_records_list = filtered_records_list[:extracted_limit]

    # Step 11: Format response output string deterministically
    output_lines = []
    modality_header = modality if modality != "all" else "file"
    if date_label:
        header_text = f"{modality_header.capitalize()}s added {date_label}:"
    else:
        header_text = f"Recently added {modality_header}s:"
    output_lines.append(header_text)

    for item_index, record_item in enumerate(filtered_records_list, start=1):
        formatted_date_string = record_item["datetime"].strftime("%B %d, %Y, %H:%M")
        output_lines.append(f"{item_index}. {record_item['filename']} — {formatted_date_string}")

    answer_string = "\n".join(output_lines)
    source_filenames = [record["filename"] for record in filtered_records_list]
    matched_chunks_count = len(filtered_records_list)

    if DEBUG:
        print("\n===== TEMPORAL QUERY ROUTING =====")
        print(f"Detected temporal intent: {temporal_intent}")
        print(f"Modality filter: {modality}")
        print(f"Date label filter: {date_label}")
        print(f"Limit applied: {extracted_limit}")
        print(f"Matching files count: {matched_chunks_count}")

    return answer_string, source_filenames, matched_chunks_count




def is_image_list_query(question):
    """
    Checks if query is asking to list all indexed images.
    """
    question_lower = question.lower()
    phrases = [
        "list all images", "list out all images", "list the images",
        "what images do you have", "which images do you have",
        "show all images", "show me all images", "all images"
    ]
    return any(phrase in question_lower for phrase in phrases)


def clean_llm_answer(raw_answer):
    """
    Strips accidental Markdown image tags, Source lines, and Path lines from LLM answer.
    """
    clean_text = raw_answer
    clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", clean_text)
    lines = clean_text.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Source:") or stripped.startswith("Path:"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def is_filename_stem_match(target_token, filename):
    """
    Checks if a filename matches a target entity token (e.g., 'dense' matches 'dense.jpeg', 'dense2.webp', 'dense_2.png').
    """
    if not target_token or not filename:
        return False
    
    stem = os.path.splitext(filename)[0].lower()
    target = target_token.lower()
    
    if stem == target:
        return True
    
    pattern = r"^" + re.escape(target) + r"[\s_\-]?\d*$"
    if re.match(pattern, stem):
        return True
    
    from app.query.query_analyzer import compact_alphanumeric
    compact_target = compact_alphanumeric(target)
    compact_stem = compact_alphanumeric(stem)
    if compact_stem.startswith(compact_target):
        suffix = compact_stem[len(compact_target):]
        if not suffix or suffix.isdigit():
            return True

    return False


def handle_metadata_query(question, analysis):
    """
    Handles deterministic metadata queries without calling Ollama or the LLM.
    Supports:
    - FILE_METADATA: chunk counts, file metadata
    - IMAGE_FILENAME_QUERY: returns image filenames matching pattern
    - IMAGE_COUNT_QUERY: returns count of image files matching pattern
    """
    intent = analysis.get("intent")
    source_hint = analysis.get("source_hint")
    canonical_source_id = analysis.get("canonical_source_id")

    if intent in ["FILE_METADATA", "file_count"] or "chunk" in question.lower():
        target_path = canonical_source_id if canonical_source_id else source_hint
        if not target_path:
            from app.query.query_analyzer import get_indexed_filenames, find_best_matching_source
            indexed_files = get_indexed_filenames()
            target_file, _ = find_best_matching_source(question.lower(), indexed_files)
        else:
            target_file = os.path.basename(target_path)

        if not target_file:
            return "File metadata not found in LanceDB.", [], 0

        from app.storage.lancedb_store import get_table
        doc_table = get_table()
        chunk_count = 0
        if doc_table is not None and doc_table.count_rows() > 0:
            df_doc = doc_table.to_pandas()
            if not df_doc.empty and "path" in df_doc.columns:
                target_base = target_file.lower()
                matching = df_doc[df_doc["path"].apply(lambda p: os.path.basename(str(p)).lower() == target_base)]
                chunk_count = len(matching)

        response_text = f"{target_file} has {chunk_count} indexed chunks."
        if DEBUG:
            print("\n===== QUERY INTENT =====")
            print(f"Intent: {intent}")
            print("\n===== SOURCE RESOLUTION =====")
            print(f"Requested source: {source_hint}")
            print(f"Resolved source: {target_file}")
            print(f"Canonical path: {target_path}")
            print("Resolution method: deterministic_metadata")
            print("\n===== METADATA QUERY EXECUTION =====")
            print(f"Chunk count: {chunk_count}")
        return response_text, [target_file], chunk_count, {"type": "file_metadata", "chunk_count": chunk_count, "target_file": target_file}

    elif intent in ["IMAGE_FILENAME_QUERY", "image_filename_query"]:
        from app.query.query_analyzer import get_indexed_filenames, extract_entity_tokens
        indexed_files = get_indexed_filenames()
        image_exts = (".jpg", ".jpeg", ".png", ".webp")
        indexed_images = [f for f in indexed_files if f.lower().endswith(image_exts)]

        entity_tokens = extract_entity_tokens(question.lower())
        matched_images = []

        if entity_tokens:
            target_token = entity_tokens[0]
            for img in indexed_images:
                if is_filename_stem_match(target_token, img):
                    matched_images.append(img)
        else:
            matched_images = indexed_images

        matched_images.sort()
        if matched_images:
            formatted_list = []
            for idx, img_name in enumerate(matched_images, 1):
                formatted_list.append(f"{idx}. {img_name}")
            response_text = "\n".join(formatted_list)
        else:
            response_text = f"No images found matching '{entity_tokens[0] if entity_tokens else ''}'."

        if DEBUG:
            print("\n===== QUERY INTENT =====")
            print(f"Intent: {intent}")
            print("\n===== METADATA QUERY EXECUTION =====")
            print(f"Matched filenames: {matched_images}")
        return response_text, matched_images, len(matched_images), {"type": "image_metadata", "matched": matched_images}

    elif intent in ["IMAGE_COUNT_QUERY", "image_count_query"]:
        from app.query.query_analyzer import get_indexed_filenames, extract_entity_tokens
        indexed_files = get_indexed_filenames()
        image_exts = (".jpg", ".jpeg", ".png", ".webp")
        indexed_images = [f for f in indexed_files if f.lower().endswith(image_exts)]

        entity_tokens = extract_entity_tokens(question.lower())
        matched_images = []

        if entity_tokens:
            target_token = entity_tokens[0]
            for img in indexed_images:
                if is_filename_stem_match(target_token, img):
                    matched_images.append(img)
        else:
            matched_images = indexed_images

        matched_images.sort()
        count = len(matched_images)
        response_text = f"{count}" if "how many" in question.lower() else f"Found {count} image(s) matching criteria."

        if DEBUG:
            print("\n===== QUERY INTENT =====")
            print(f"Intent: {intent}")
            print("\n===== METADATA QUERY EXECUTION =====")
            print(f"Count: {count}")
            print(f"Matched filenames: {matched_images}")
        return response_text, matched_images, count, {"type": "image_metadata", "count": count, "matched": matched_images}

    return "No metadata query matching criteria.", [], 0, {}


def handle_speaker_query(question, analysis):
    """
    Handles speaker and audio participant queries without inventing identities or returning summaries.
    """
    source_hint = analysis.get("source_hint")
    canonical_source_id = analysis.get("canonical_source_id")
    target_file = os.path.basename(canonical_source_id) if canonical_source_id else (source_hint if source_hint else "audio file")

    question_lower = question.lower()
    if any(phrase in question_lower for phrase in ["lady", "woman", "female", "girl"]):
        response_text = "I can't reliably determine whether a woman is speaking from the indexed transcript because speaker gender/diarization information is not available."
    else:
        response_text = "The indexed transcript does not contain reliable speaker diarization, so I cannot determine the exact number of speakers."

    if DEBUG:
        print("\n===== QUERY INTENT =====")
        print(f"Intent: {analysis.get('intent')}")
        print("\n===== AUDIO SPEAKER QUERY EXECUTION =====")
        print(f"Target file: {target_file}")
        print(f"Response: {response_text}")
    return response_text, [target_file] if source_hint else [], 1


def translate_full_transcript(full_transcript, target_language="English", source_filename=""):
    """
    Faithful full-transcript translation service using sequential batch processing.
    Splits long transcripts into sequential paragraph/line batches (~1000 chars each),
    translates each batch verbatim into target_language, and recombines in exact original order.
    Enforces 18 strict translation rules preventing summarization, interpretation, or silent ASR correction.
    Returns tuple: (translated_transcript, translated_batch_count)
    """
    if not full_transcript or not full_transcript.strip():
        return "", 0

    target_lang_display = target_language.capitalize() if target_language else "English"
    source_label = f"Source File: {os.path.basename(source_filename)}\n" if source_filename else ""

    # Split transcript into logical paragraph/line blocks for batch processing
    raw_lines = full_transcript.split("\n")
    batches = []
    current_batch_lines = []
    current_length = 0
    max_batch_chars = 4000

    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue

        if current_length + len(line_str) + 1 <= max_batch_chars:
            current_batch_lines.append(line_str)
            current_length += len(line_str) + 1
        else:
            if current_batch_lines:
                batches.append("\n".join(current_batch_lines))
            current_batch_lines = [line_str]
            current_length = len(line_str)

    if current_batch_lines:
        batches.append("\n".join(current_batch_lines))

    translated_batches = []
    translated_batch_count = len(batches)

    system_instruction = (
        f"You are a strictly verbatim transcript translator translating into {target_lang_display}. "
        "TRANSLATION ONLY, not summarization, interpretation, correction, or reconstruction. "
        "1. Do NOT summarize. 2. Do NOT interpret. 3. Do NOT correct ASR errors. "
        "4. Do NOT invent missing words. 5. Do NOT infer or correct names. 6. Do NOT infer or correct numbers. "
        "7. Do NOT convert ambiguous speech into a plausible statement. 8. Translate understandable Hindi/Hinglish to English. "
        "9. Preserve original meaning. 10. Preserve uncertainty if phrase is unclear. "
        "11. Preserve names, numbers, percentages, dates, quantities, and claims. 12. Preserve original sequence/order. "
        "13. Do NOT omit repetitive or noisy content. 14. Do NOT merge segments so information disappears. "
        "15. Every input segment must have corresponding translated content. 16. No added explanations or comments. "
        "17. Do not use outside knowledge. 18. RAW ASR IS THE SOURCE OF TRUTH."
    )

    for idx, batch_text in enumerate(batches, 1):
        prompt = f"""Translate the provided ASR transcript segment into {target_lang_display}.

IMPORTANT:
This is TRANSLATION ONLY, not summarization, interpretation, correction, or reconstruction.

RULES:
1. Do NOT summarize.
2. Do NOT interpret.
3. Do NOT correct ASR errors.
4. Do NOT invent missing words.
5. Do NOT infer or correct names.
6. Do NOT infer or correct numbers.
7. Do NOT convert ambiguous speech into a more plausible statement.
8. Translate understandable Hindi/Hinglish into {target_lang_display}.
9. Preserve the original meaning as closely as possible.
10. If a phrase is unclear in the source, preserve its uncertainty rather than guessing its intended meaning.
11. Preserve names, numbers, percentages, dates, quantities, repeated statements, and factual claims.
12. Preserve the original sequence/order of the transcript.
13. Do NOT omit any content because it appears repetitive, noisy, grammatically incorrect, or meaningless.
14. Do NOT merge separate source segments in a way that causes information to disappear.
15. Every input segment must have corresponding translated content.
16. Do not add explanations, comments, corrections, or assumptions.
17. Do not use outside knowledge to repair or interpret the ASR.
18. If the source contains an unclear/noisy phrase, translate whatever is understandable and preserve the uncertainty of the remaining part.

RAW ASR IS THE SOURCE OF TRUTH.

{source_label}RAW BATCH TRANSCRIPT (Segment {idx}/{len(batches)}):
{batch_text}

FAITHFUL {target_lang_display.upper()} TRANSLATION:"""

        try:
            raw_translated_batch = ask_gemini(prompt)
        except Exception:
            try:
                raw_translated_batch = ask_llama(prompt, system_instruction=system_instruction)
            except Exception as e:
                raw_translated_batch = f"Translation Error: {str(e)}"

        clean_batch = clean_llm_answer(raw_translated_batch)
        if clean_batch:
            translated_batches.append(clean_batch)

    final_translated_transcript = "\n".join(translated_batches).strip()

    if DEBUG:
        print("\n===== TRANSLATION BATCH METRICS =====")
        print(f"source_filename: {os.path.basename(source_filename)}")
        print(f"target_language: {target_lang_display}")
        print(f"translated_batch_count: {translated_batch_count}")
        print(f"input_char_length: {len(full_transcript)}")
        print(f"output_char_length: {len(final_translated_transcript)}")

    return final_translated_transcript, translated_batch_count


def ask(question, return_structured=False):
    """
    Main Multimodal RAG Orchestrator function.
    Extracts QueryPlan and dispatches execution to IntentRouter.
    """
    from app.rag.intent_router import IntentRouter
    from app.services.face_service import (
        analyze_faces_in_image,
        register_pending_face,
        search_images_by_registered_face,
        FACE_COSINE_DISTANCE_THRESHOLD
    )
    from app.services.interaction_state import (
        get_pending_faces,
        set_pending_faces,
        clear_pending_faces
    )

    question_lower = question.lower().strip()

    # Step 1: Query Analysis & QueryPlan Construction
    analysis = analyze_query(question)
    plan = analysis.get("plan")

    if DEBUG:
        print("\n===== QUERY PLAN =====")
        print(f"Question: {question}")
        print(f"Intent: {plan.intent.value if plan else analysis.get('intent')}")
        print(f"Scope: {plan.scope.value if plan else analysis.get('request_scope')}")
        print(f"Modality: {plan.modality.value if plan else analysis.get('modality')}")
        print(f"Source Hint: {plan.source_spec.source_hint if plan else analysis.get('source_hint')}")
        print(f"Canonical Path: {plan.source_spec.canonical_path if plan else analysis.get('canonical_source_id')}")

    # Check for pending face registration flow first
    pending_faces = get_pending_faces()
    if pending_faces and (analysis.get("face_intent") == "face_registration" or any(p in question_lower for p in ["this is", "my mother", "her name", "his name"])):
        user_label = question.strip()
        for prefix in ["this is my ", "this is ", "her name is ", "his name is "]:
            if question_lower.startswith(prefix):
                user_label = question[len(prefix):].strip(" .!")
                break

        if not user_label:
            user_label = "mother"

        exact_pending_face = pending_faces[0]
        reg_result = register_pending_face(exact_pending_face, user_label)
        clear_pending_faces()

        ans_str = f"Registered identity '{user_label}' in face memory."
        sources = [os.path.basename(exact_pending_face.get("source_id", "image.jpg"))]
        update_last_interaction(question, ans_str, sources, "image", identity_info={"registered_label": user_label})

        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": 1, "type": "text", "images": []}
        return ans_str, sources, 1

    # Step 2: Dispatch QueryPlan to IntentRouter strategy handler
    if plan:
        return IntentRouter.dispatch(plan, question, analysis, return_structured=return_structured)

    ans_str, sources, num_chunks = handle_temporal_query(question, analysis)
    if return_structured:
        return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "type": "text", "images": []}
    return ans_str, sources, num_chunks


    # Check for Deterministic Metadata Queries (No LLM call!)
    if intent in ["FILE_METADATA", "IMAGE_FILENAME_QUERY", "IMAGE_COUNT_QUERY", "IMAGE_FILENAME_QUERY", "image_filename_query", "image_count_query"]:
        res_meta = handle_metadata_query(question, analysis)
        if isinstance(res_meta, tuple) and len(res_meta) == 4:
            ans_str, sources, num_chunks, meta_info = res_meta
        else:
            ans_str, sources, num_chunks = res_meta[0], res_meta[1], res_meta[2]
            meta_info = {}
        update_last_interaction(question, ans_str, sources, modality)
        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "metadata_info": meta_info, "type": "text", "images": []}
        return ans_str, sources, num_chunks

    # Check for Speaker Queries
    if intent in ["AUDIO_SPEAKER_QUERY", "speaker_analysis"]:
        ans_str, sources, num_chunks = handle_speaker_query(question, analysis)
        update_last_interaction(question, ans_str, sources, modality)
        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "type": "text", "images": []}
        return ans_str, sources, num_chunks

    # Resolve target source filepath if canonical_source_id is available
    resolved_source_path = canonical_source_id if canonical_source_id else source_hint
    source_exists = bool(resolved_source_path and os.path.exists(resolved_source_path))

    # Handle unresolved explicit source query cleanly without falling back to unrelated files
    unresolved_explicit_source = analysis.get("unresolved_explicit_source", False)
    if (source_hint and str(source_hint).startswith("UNRESOLVED_")) or unresolved_explicit_source:
        clean_name = str(source_hint).replace("UNRESOLVED_SOURCE_", "").replace("UNRESOLVED_AUDIO_SOURCE", "").strip()
        target_display = clean_name.capitalize() if clean_name else "requested"
        unresolved_msg = f"I couldn't reliably identify the requested {target_display} file, so I won't use another file to answer this question."
        
        if DEBUG:
            print("\n===== SOURCE FILTER =====")
            print(f"Source filter: UNRESOLVED ({source_hint}) -> 0 candidates retrieved. Preventing fallback to unrelated files.")
        
        update_last_interaction(question, unresolved_msg, [], modality)
        if return_structured:
            return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
        return unresolved_msg, [], 0


    # Check for pending face registration flow first
    pending_faces = get_pending_faces()
    if pending_faces and (face_intent == "face_registration" or any(p in question_lower for p in ["this is", "my mother", "her name", "his name"])):
        user_label = question.strip()
        for prefix in ["this is my ", "this is ", "her name is ", "his name is "]:
            if question_lower.startswith(prefix):
                user_label = question[len(prefix):].strip(" .!")
                break

        if not user_label:
            user_label = "mother"

        exact_pending_face = pending_faces[0]
        reg_result = register_pending_face(exact_pending_face, user_label)
        clear_pending_faces()

        ans_str = f"Registered identity '{user_label}' in face memory."
        sources = [os.path.basename(exact_pending_face.get("source_id", "image.jpg"))]
        update_last_interaction(question, ans_str, sources, "image", identity_info={"registered_label": user_label})

        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": 1, "type": "text", "images": []}
        return ans_str, sources, 1

    # Step 2: Handle Follow-up Correction Intent
    if not is_correction and last_state.get("previous_query"):
        if any(phrase in question_lower for phrase in ["but there are", "there are two", "one is", "actually", "wrong"]):
            is_correction = True
            intent = "correction"

    preferred_sources = []
    rejected_sources = []

    if is_correction:
        prev_query = last_state.get("previous_query", "")
        prev_answer = last_state.get("previous_answer", "")
        prev_source = last_state.get("previous_source", "")
        prev_modality = last_state.get("previous_modality", "all")
        retrieved_sources = last_state.get("retrieved_sources", [])

        wrong_source = prev_source if prev_source else (retrieved_sources[0] if retrieved_sources else "")
        correct_source = source_hint if source_hint else (prev_source if prev_source else "mummy.jpg")

        if correct_source and wrong_source and correct_source.lower() == wrong_source.lower():
            wrong_source = ""

        corrected_ans = question
        target_modality = modality if modality != "all" else (prev_modality if prev_modality != "all" else "image")

        store_feedback(
            original_query=prev_query if prev_query else question,
            wrong_answer=prev_answer,
            wrong_source=wrong_source,
            correct_source=correct_source,
            corrected_answer=corrected_ans,
            correction_text=question,
            modality=target_modality,
            rejected_sources=wrong_source,
            confidence=1.0,
            feedback_type="answer_correction"
        )

        learning_answer = (
            f"LEARNING MODE ACTIVATED: Correction stored in feedback memory for query '{prev_query}'. "
            f"Source set to '{correct_source}'."
        )

        update_last_interaction(question, learning_answer, [correct_source], target_modality, identity_info={"correction": question})

        if return_structured:
            return {"answer": learning_answer, "sources": [correct_source], "num_chunks": 1, "type": "text", "images": []}
        return learning_answer, [correct_source], 1

    request_scope = analysis.get("request_scope", "selective")

    # Step 3: Handle Audio/Document Full Transcript & Translation Intent
    if intent in ["AUDIO_TRANSCRIPT", "AUDIO_TRANSLATION", "transcript", "full_transcript"] and request_scope == "complete_file":
        if not resolved_source_path or not os.path.exists(resolved_source_path):
            target_display = os.path.basename(source_hint) if source_hint else "requested"
            unresolved_msg = f"I couldn't reliably identify the requested {target_display} file, so I won't use another file to answer this question."
            update_last_interaction(question, unresolved_msg, [], modality)
            if return_structured:
                return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return unresolved_msg, [], 0

        full_transcript, total_chunks, val_metrics = get_full_transcript_for_source(resolved_source_path)
        src_name = os.path.basename(resolved_source_path)

        if not val_metrics.get("is_complete", True) and val_metrics.get("missing_chunks", 0) > 0:
            incomplete_msg = (
                f"Source file '{src_name}' could not be fully reconstructed. "
                f"Expected {val_metrics.get('expected_chunks', 0)} chunks, but fetched {val_metrics.get('fetched_chunks', 0)} chunks."
            )
            update_last_interaction(question, incomplete_msg, [src_name], modality)
            if return_structured:
                return {
                    "answer": incomplete_msg,
                    "sources": [src_name],
                    "num_chunks": val_metrics.get("fetched_chunks", 0),
                    "type": "text",
                    "images": [],
                    "validation_metrics": val_metrics
                }
            return incomplete_msg, [src_name], val_metrics.get("fetched_chunks", 0)

        if full_transcript:
            if intent == "AUDIO_TRANSLATION" or target_language is not None:
                target_lang_name = target_language if target_language else "English"
                final_output, translated_batch_count = translate_full_transcript(
                    full_transcript,
                    target_language=target_lang_name,
                    source_filename=resolved_source_path
                )
            else:
                final_output = full_transcript
                translated_batch_count = 1

            validate_content_grounding(full_transcript, final_output, question, resolved_source_path)

            if DEBUG:
                print("\n===== FULL TRANSCRIPT PIPELINE METRICS =====")
                print(f"Source: {src_name}")
                print(f"expected_chunk_count: {val_metrics.get('expected_chunks', total_chunks)}")
                print(f"retrieved_chunk_count: {val_metrics.get('fetched_chunks', total_chunks)}")
                print(f"missing_chunks: {val_metrics.get('missing_chunks', 0)}")
                print(f"duplicate_chunks: {val_metrics.get('duplicate_chunks', 0)}")
                print(f"reconstructed_chunk_count: {val_metrics.get('reconstructed_chunks', total_chunks)}")
                print(f"translated_batch_count: {translated_batch_count}")
                print(f"Intent: {intent}")
                print(f"Target language: {target_language}")
                print(f"Reconstructed raw length: {len(full_transcript)} chars")
                print(f"Final output length: {len(final_output)} chars")

            update_last_interaction(question, final_output, [src_name], modality)
            if return_structured:
                return {
                    "answer": final_output,
                    "sources": [src_name],
                    "num_chunks": total_chunks,
                    "evidence": [{"source": src_name, "chunk_id": 1, "text": full_transcript}],
                    "type": "text",
                    "images": [],
                    "validation_metrics": val_metrics
                }
            return final_output, [src_name], total_chunks

    # Check for explicit Image OCR intent
    if intent in ["IMAGE_OCR", "image_ocr"] and resolved_source_path and os.path.exists(resolved_source_path):
        ocr_text, total_chunks = get_stored_ocr_text_for_image(resolved_source_path)
        src_name = os.path.basename(resolved_source_path)

        if DEBUG:
            print("\n===== IMAGE DEBUG =====")
            print(f"Source: {src_name}")
            print(f"OCR length: {len(ocr_text)} chars")

        if ocr_text and ocr_text.strip():
            update_last_interaction(question, ocr_text, [src_name], "image")
            if return_structured:
                return {"answer": ocr_text, "sources": [src_name], "num_chunks": total_chunks, "ocr_text": ocr_text, "type": "text", "images": []}
            return ocr_text, [src_name], total_chunks

    # Check for Image Routing Branches (Source Image Retrieval, Face Verification, Face Memory Search)
    is_image_source_resolved = bool(source_exists and modality == "image")
    person_name = extract_person_name_from_question(question)
    source_stem = os.path.splitext(os.path.basename(source_hint))[0].lower() if source_hint else ""

    is_person_verification_query = False
    if is_image_source_resolved:
        colloquial_synonyms = {
            "mummy": ["mummy", "mom", "mother"],
            "mom": ["mummy", "mom", "mother"],
            "mother": ["mummy", "mom", "mother"]
        }
        related_synonyms = colloquial_synonyms.get(source_stem, [source_stem])

        if face_intent in ["face_identification", "face_verification"]:
            is_person_verification_query = True
        elif person_name and person_name.lower() not in related_synonyms:
            is_person_verification_query = True

    is_explicit_image_retrieval = (intent == "image_retrieval") or is_image_list_query(question) or any(
        p in question_lower for p in ["show image", "find image", "get image", "list images", "show me"]
    ) and not any(w in question_lower for w in ["what is in", "content of", "summarize", "text", "names in"])

    # ROUTE BRANCH 1: Source-Based Image Retrieval
    if is_image_source_resolved and is_explicit_image_retrieval and not is_person_verification_query and not is_visual_qa:
        if DEBUG:
            print("\n===== IMAGE ROUTING =====")
            print("Route: source_image_retrieval")
        src_name = os.path.basename(resolved_source_path)
        ans_str = f"Retrieved image source '{src_name}'."
        update_last_interaction(question, ans_str, [src_name], "image")

        if return_structured:
            return {
                "answer": ans_str,
                "sources": [src_name],
                "num_chunks": 1,
                "type": "image",
                "images": [{"type": "image", "path": resolved_source_path, "source": src_name}]
            }
        return ans_str, [src_name], 1

    # ROUTE BRANCH 2: Face Verification inside Specific Image Source
    elif is_image_source_resolved and is_person_verification_query:
        if DEBUG:
            print("\n===== IMAGE ROUTING =====")
            print("Route: face_verification")
        target_img = resolved_source_path
        face_res = analyze_faces_in_image(target_img)
        faces_detected = face_res.get("faces_detected", 0)
        faces_list = face_res.get("faces", [])
        src_name = os.path.basename(target_img)

        if person_name and person_name.lower() != source_stem:
            matching_face = None
            for f in faces_list:
                if f.get("person_name") and f.get("person_name").lower() == person_name.lower() and f.get("status") == "known":
                    matching_face = f
                    break

            if matching_face:
                msg = f"Yes, {person_name} was found in {src_name}."
                update_last_interaction(question, msg, [src_name], "image")
                if return_structured:
                    return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
                return msg, [src_name], len(faces_list)
            else:
                msg = f"No registered face memory record for '{person_name}' was found in {src_name}."
                update_last_interaction(question, msg, [src_name], "image")
                if return_structured:
                    return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
                return msg, [src_name], len(faces_list)

        if faces_detected == 0:
            msg = "I couldn't detect a usable face in this image."
            update_last_interaction(question, msg, [src_name], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_name], "num_chunks": 0, "type": "text", "images": []}
            return msg, [src_name], 0

        known_names = [f["person_name"] for f in faces_list if f.get("person_name")]
        msg = f"Identified person: {', '.join(known_names)}." if known_names else "I found a person I don't recognize yet. Who is this person?"
        update_last_interaction(question, msg, [src_name], "image")
        if return_structured:
            return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
        return msg, [src_name], len(faces_list)

    # ROUTE BRANCH 3: Persistent Face-Memory Search
    elif (face_intent == "face_search" or is_face_search_query(question)) and not is_image_source_resolved:
        if DEBUG:
            print("\n===== IMAGE ROUTING =====")
            print("Route: face_memory_search")
        if person_name:
            matched_records = search_images_by_registered_face(person_name)
            if not matched_records:
                msg = f"No registered face memory record found matching '{person_name}'."
                update_last_interaction(question, msg, [], "image")
                if return_structured:
                    return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
                return msg, [], 0

            image_paths = sorted(list(set(r.get("image_path") or r.get("path") for r in matched_records if (r.get("image_path") or r.get("path")))))
            sources = [os.path.basename(p) for p in image_paths]
            ans_str = f"Found {len(sources)} image(s) matching registered face memory for '{person_name}'."
            update_last_interaction(question, ans_str, sources, "image")

            if return_structured:
                return {"answer": ans_str, "sources": sources, "num_chunks": len(sources), "type": "image", "images": [{"type": "image", "path": p, "source": os.path.basename(p)} for p in image_paths]}
            return ans_str, sources, len(sources)

    # Step 4: Handle Visual Question Answering & Image Summary/QA
    if intent in ["IMAGE_SUMMARY", "IMAGE_VISUAL_QUERY", "image_summary"] or (modality == "image" and source_hint):
        target_img = resolved_source_path
        if not target_img or not os.path.exists(target_img):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            c_path = os.path.join(base_dir, "watched_folder", source_hint if source_hint else "mummy.jpg")
            if os.path.exists(c_path):
                target_img = c_path
            else:
                target_img = os.path.join(base_dir, source_hint if source_hint else "mummy.jpg")

        if not os.path.exists(target_img):
            msg = f"No usable visual information was found for {source_hint}."
            update_last_interaction(question, msg, [source_hint] if source_hint else [], "image")

            if return_structured:
                return {"answer": msg, "sources": [source_hint] if source_hint else [], "num_chunks": 0, "type": "text", "images": []}
            return msg, [source_hint] if source_hint else [], 0

        # Retrieve stored clean OCR text first
        ocr_text, ocr_chunks = get_stored_ocr_text_for_image(target_img)
        src_name = os.path.basename(target_img)

        if DEBUG:
            print("\n===== IMAGE DEBUG =====")
            print(f"Source: {src_name}")
            print(f"OCR length: {len(ocr_text)} chars")


        if ocr_text and ocr_text.strip():
            vqa_prompt = f"""You are a Visual Assistant. Analyze the image content below and fulfill the user request.

IMAGE OCR EVIDENCE:
{ocr_text.strip()}

USER REQUEST: {question}

RESPONSE FORMAT INSTRUCTIONS:
Separate your answer into:
A. Directly visible/readable text: List exact words and labels visible.
B. Visual description: Describe the diagram or image structure.
C. Interpretation: Provide a cautious explanation, preserving any uncertainty rather than inventing facts.

Response:"""
            try:
                raw_vqa = ask_llama(vqa_prompt)
            except Exception:
                try:
                    raw_vqa = ask_gemini(vqa_prompt)
                except Exception as e:
                    raw_vqa = f"Content in {src_name}:\n{ocr_text.strip()}"

            vqa_answer = clean_llm_answer(raw_vqa)
        else:
            vqa_answer = summarize_image(target_img, question)
            refusal_phrases = [
                "cannot provide", "no image provided", "not provided an image",
                "have not provided", "haven't provided", "please upload", "without being able to see"
            ]
            if any(rp in vqa_answer.lower() for rp in refusal_phrases):
                vqa_answer = f"Visual details for image '{src_name}': The image file is stored and indexed in LanceDB."

        update_last_interaction(question, vqa_answer, [src_name], "image")

        if return_structured:
            return {"answer": vqa_answer, "sources": [src_name], "num_chunks": 1, "type": "text", "images": []}
        return vqa_answer, [src_name], 1

    # Step 5: System A Hard Source Candidate Retrieval
    if modality == "audio" and resolved_source_path and os.path.exists(resolved_source_path):
        full_transcript, total_chunks, _ = get_full_transcript_for_source(resolved_source_path)
        if full_transcript:
            src_name = os.path.basename(resolved_source_path)
            sources = [src_name]
            context_str = full_transcript
            num_chunks = total_chunks
        else:
            sources = []
            context_str = ""
            num_chunks = 0
    else:
        feedback_candidates = search_feedback(question, query_modality=modality, source_hint=source_hint, limit=5)
        if feedback_candidates:
            for fb in feedback_candidates:
                c_src = fb.get("correct_source")
                w_src = fb.get("wrong_source")
                if c_src and c_src not in preferred_sources:
                    if not source_hint or c_src.lower() == source_hint.lower() or os.path.basename(c_src).lower() == os.path.basename(source_hint).lower():
                        preferred_sources.append(c_src)
                if w_src and w_src not in rejected_sources:
                    rejected_sources.append(w_src)

        if source_hint and source_hint not in preferred_sources:
            preferred_sources.append(source_hint)

        results = search(
            question,
            max_results=10,
            analysis=analysis,
            preferred_sources=preferred_sources,
            rejected_sources=rejected_sources
        )

        if not results:
            target_source = source_hint if source_hint else "the requested file"
            is_img_source = (modality == "image") or (target_source and target_source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
            no_res_msg = f"No usable visual information was found for {target_source}." if is_img_source else f"No relevant content was found in {target_source}."
            update_last_interaction(question, no_res_msg, [target_source], modality)

            if return_structured:
                return {"answer": no_res_msg, "sources": [target_source], "num_chunks": 0, "type": "text", "images": []}
            return no_res_msg, [target_source], 0

        # Collect sources & context
        retrieved_sources_set = set()
        context_list = []
        for doc in results:
            fname = os.path.basename(doc["path"])
            retrieved_sources_set.add(fname)
            context_list.append(doc["text"])

        sources = sorted(list(retrieved_sources_set))
        context_str = "\n\n".join(context_list)
        num_chunks = len(results)

    # Step 6: Build Grounded Prompt for LLM (NO FAKE NAME INJECTIONS)
    source_context_label = f"Source File: {os.path.basename(resolved_source_path)}\n" if resolved_source_path else ""

    # Detect user question language for Language Control
    is_hindi_question = any("\u0900" <= c <= "\u097f" for c in question) or any(
        hw in question_lower for hw in ["kya", "baatein", "hui", "hai", "kaun", "batao", "bataiye", "call me", "me kya"]
    )
    
    if is_hindi_question:
        translated_hint = "What was discussed in this audio recording / call?" if any(w in question_lower for w in ["baatein", "discuss", "hua", "summary", "con call"]) else question
        lang_instruction = (
            f"The user asked in Hindi/Hinglish: '{question}' (Meaning: '{translated_hint}'). "
            "Provide a clear, helpful, and strictly grounded summary/answer in natural Hindi or Hinglish based ONLY on facts in the retrieved context."
        )
    else:
        lang_instruction = "Respond in clear, professional English."

    prompt = f"""Answer the user's question using ONLY the retrieved context below.

STRICT GROUNDING RULES:
1. Rely strictly on facts explicitly stated in the retrieved context. Never invent meanings, dates, numbers, company names, family names, or personal names.
2. Identify topics, people, and events directly from the text.
3. If the user asks for names, aliases, or background of a person or entity, list ONLY the exact names explicitly stated in the context text. Do NOT invent full names, father names, grandfather names, or aliases from external world knowledge.
4. If the transcript or context text contains noisy ASR words or unclear names, preserve the raw transcript wording or state that the name is unclear rather than guessing fictitious company or personal names.
5. {lang_instruction}

{source_context_label}Retrieved Context:
{context_str}

Question: {question}

Answer:"""

    grounded_sys_instruction = (
        "You are a strictly grounded Personal Second Brain Assistant. "
        "Answer ONLY using facts present in the retrieved context. "
        "Do NOT invent concepts, dates, numbers, company names, father names, or personal face identities."
    )

    try:
        raw_answer = ask_gemini(prompt)
    except Exception:
        try:
            raw_answer = ask_llama(prompt, system_instruction=grounded_sys_instruction)
        except Exception as e:
            raw_answer = f"LLM Error: {str(e)}"

    clean_answer = clean_llm_answer(raw_answer)

    # Step 7: Content Grounding Validation
    validate_content_grounding(context_str, clean_answer, question, resolved_source_path)

    update_last_interaction(question, clean_answer, sources, modality)

    if return_structured:
        evidence_list = []
        if 'results' in locals() and results:
            for idx, doc in enumerate(results, 1):
                evidence_list.append({
                    "source": os.path.basename(doc.get("path", "")),
                    "chunk_id": doc.get("chunk_id", idx),
                    "timestamp": doc.get("timestamp", ""),
                    "text": doc.get("text", "")
                })
        elif 'full_transcript' in locals() and full_transcript:
            evidence_list.append({
                "source": os.path.basename(resolved_source_path) if resolved_source_path else "Audio",
                "chunk_id": 1,
                "text": full_transcript
            })
        return {
            "answer": clean_answer,
            "sources": sources,
            "num_chunks": num_chunks,
            "evidence": evidence_list,
            "type": "text",
            "images": []
        }

    return clean_answer, sources, num_chunks




def summarize(question, context):
    """
    Summarization helper function using strict grounding rules.
    """
    prompt = f"Summarize the retrieved context below.\n\nRetrieved Context:\n{context}\n\nUser Request:\n{question}\n\nSummary:"
    grounded_sys_instruction = (
        "You are a strictly grounded Personal Second Brain Assistant. "
        "Summarize ONLY using facts present in the retrieved context. "
        "Do NOT invent missing facts, inventory, purchases, or accounting terms."
    )
    try:
        return clean_llm_answer(ask_gemini(prompt))
    except Exception:
        try:
            return clean_llm_answer(ask_llama(prompt, system_instruction=grounded_sys_instruction))
        except Exception as e:
            return f"LLM Error: {str(e)}"


def format_response(arg1, arg2=None, arg3=None, arg4=0, evidence=None):
    """
    Formats the clean CLI terminal output according to user-facing specifications.
    Outputs exclusively:
    - USER QUERY
    - ANSWER
    - SOURCE
    - NUMBER OF RELEVANT CHUNKS
    """
    question = ""
    answer = ""
    sources = []
    num_chunks = 0

    if isinstance(arg2, dict):
        question = arg1 if isinstance(arg1, str) else ""
        answer = arg2.get("answer", "")
        sources = arg2.get("sources", [])
        num_chunks = arg2.get("num_chunks", 0)
    elif isinstance(arg2, (list, tuple)) or (isinstance(arg1, str) and not isinstance(arg2, str) and arg2 is not None):
        question = ""
        answer = arg1
        sources = arg2 if arg2 is not None else []
        num_chunks = arg3 if arg3 is not None else 0
    else:
        question = arg1 if isinstance(arg1, str) else ""
        answer = arg2 if arg2 is not None else ""
        sources = arg3 if arg3 is not None else []
        num_chunks = arg4 if arg4 is not None else 0

    header = "=" * 60
    sections = []

    # 1. USER QUERY
    if question and question.strip():
        sections.append(f"{header}\nUSER QUERY\n{header}\n{question.strip()}")

    # 2. ANSWER
    sections.append(f"{header}\nANSWER\n{header}\n{answer.strip() if answer else 'No answer generated.'}")

    # 3. SOURCE
    if sources:
        source_str = ", ".join([os.path.basename(s) for s in sources])
    else:
        source_str = "None"
    sections.append(f"{header}\nSOURCE\n{header}\n{source_str}")

    # 4. NUMBER OF RELEVANT CHUNKS
    sections.append(f"{header}\nNUMBER OF RELEVANT CHUNKS\n{header}\n{num_chunks}")

    return "\n\n".join(sections)



if __name__ == "__main__":
    while True:
        try:
            question = input("\nAsk: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower().strip() in ["exit", "quit"]:
            break
        if not question.strip():
            continue

        res_struct = ask(question, return_structured=True)
        print("\n" + format_response(question, res_struct))