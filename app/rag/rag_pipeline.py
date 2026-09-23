import os
import re
from app.search.vector_search import search, get_full_content_for_source, get_stored_ocr_text_for_image
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


def clean_llm_answer(raw_answer):
    """
    Strips accidental Markdown image tags, Source lines, and Path lines
    from LLM answer text before returning it to the user.
    """
    # Remove markdown image tags like ![alt](url)
    clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", raw_answer)

    # Remove any lines the LLM accidentally prefixed with "Source:" or "Path:"
    lines = clean_text.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Source:") or stripped.startswith("Path:"):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def handle_temporal_query(question, analysis, plan_modality=None, plan_start_dt=None, plan_end_dt=None, plan_date_label=None, plan_limit=None):
    """Deterministic file inventory/list/count handler using the filesystem as truth."""
    from datetime import datetime
    from pathlib import Path
    from app.query.query_analyzer import INTENT_FILE_COUNT

    project_root = Path(__file__).resolve().parents[2]
    watched_folder = project_root / "watched_folder"
    if not watched_folder.is_dir():
        return "The watched folder does not exist.", [], 0

    modality = plan_modality if plan_modality is not None else analysis.get("modality", "all")
    start_datetime = plan_start_dt if plan_start_dt is not None else analysis.get("start_datetime")
    end_datetime = plan_end_dt if plan_end_dt is not None else analysis.get("end_datetime")
    date_label = plan_date_label if plan_date_label is not None else analysis.get("date_label")
    extracted_limit = plan_limit if plan_limit is not None else analysis.get("extracted_limit")
    intent = analysis.get("intent", "")
    question_lower = question.lower()

    extension_groups = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"},
        "pdf": {".pdf"},
        "docx": {".doc", ".docx"},
        "document": {".doc", ".docx", ".txt", ".md", ".rtf", ".odt"},
        "spreadsheet": {".xls", ".xlsx", ".csv", ".ods"},
        "presentation": {".ppt", ".pptx", ".odp"},
        "archive": {".zip", ".tar", ".gz", ".tgz"},
    }

    records = []
    for path in watched_folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        records.append({"filename": path.name, "path": str(path), "datetime": mtime})

    extensions = extension_groups.get(modality)
    if extensions is not None:
        records = [r for r in records if Path(r["filename"]).suffix.lower() in extensions]

    if start_datetime is not None and end_datetime is not None:
        records = [r for r in records if start_datetime <= r["datetime"] <= end_datetime]

    records.sort(key=lambda r: r["datetime"], reverse=True)

    is_count = (
        intent == INTENT_FILE_COUNT
        or bool(re.search(r"\b(?:how many|count of|number of)\b", question_lower))
    )
    label = {"image":"image", "pdf":"PDF", "docx":"document", "document":"document",
             "spreadsheet":"spreadsheet", "presentation":"presentation", "archive":"archive",
             "all":"file"}.get(modality, "file")

    if is_count:
        return f"Total {label} files: {len(records)}.", [r["filename"] for r in records], len(records)

    if not records:
        return f"No {label} files were found" + (f" for {date_label}." if date_label else "."), [], 0

    if extracted_limit is not None and extracted_limit > 0:
        records = records[:extracted_limit]

    header = f"{label.capitalize()}s" + (f" {date_label}" if date_label else "") + ":"
    lines = [header]
    for i, record in enumerate(records, 1):
        lines.append(f"{i}. {record['filename']} — {record['datetime'].strftime('%B %d, %Y, %H:%M')}")
    return "\n".join(lines), [r["filename"] for r in records], len(records)


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




def translate_full_content(full_content, target_language="English", source_filename=""):
    """
    Faithful full-content translation service using sequential batch processing.
    Splits long source content into sequential paragraph/line batches (~1000 chars each),
    translates each batch verbatim into target_language, and recombines in exact original order.
    Enforces 18 strict translation rules preventing summarization, interpretation, or silent source-content correction.
    Returns tuple: (translated_content, translated_batch_count)
    """
    if not full_content or not full_content.strip():
        return "", 0

    target_lang_display = target_language.capitalize() if target_language else "English"
    source_label = f"Source File: {os.path.basename(source_filename)}\n" if source_filename else ""

    # Split source content into logical paragraph/line blocks for batch processing
    raw_lines = full_content.split("\n")
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
        f"You are a strictly verbatim content translator translating into {target_lang_display}. "
        "TRANSLATION ONLY, not summarization, interpretation, correction, or reconstruction. "
        "1. Do NOT summarize. 2. Do NOT interpret. 3. Do NOT correct source-content errors or formatting. "
        "4. Do NOT invent missing words. 5. Do NOT infer or correct names. 6. Do NOT infer or correct numbers. "
        "7. Do NOT convert ambiguous source content into a plausible statement. 8. Translate understandable Hindi/Hinglish to English. "
        "9. Preserve original meaning. 10. Preserve uncertainty if phrase is unclear. "
        "11. Preserve names, numbers, percentages, dates, quantities, and claims. 12. Preserve original sequence/order. "
        "13. Do NOT omit repetitive or noisy content. 14. Do NOT merge segments so information disappears. "
        "15. Every input segment must have corresponding translated content. 16. No added explanations or comments. "
        "17. Do not use outside knowledge. 18. THE SOURCE CONTENT IS THE SOURCE OF TRUTH."
    )

    for idx, batch_text in enumerate(batches, 1):
        prompt = f"""Translate the provided source content segment into {target_lang_display}.

IMPORTANT:
This is TRANSLATION ONLY, not summarization, interpretation, correction, or reconstruction.

RULES:
1. Do NOT summarize.
2. Do NOT interpret.
3. Do NOT correct source-content errors or formatting.
4. Do NOT invent missing words.
5. Do NOT infer or correct names.
6. Do NOT infer or correct numbers.
7. Do NOT convert ambiguous source content into a more plausible statement.
8. Translate understandable Hindi/Hinglish into {target_lang_display}.
9. Preserve the original meaning as closely as possible.
10. If a phrase is unclear in the source, preserve its uncertainty rather than guessing its intended meaning.
11. Preserve names, numbers, percentages, dates, quantities, repeated statements, and factual claims.
12. Preserve the original sequence/order of the source content.
13. Do NOT omit any content because it appears repetitive, noisy, grammatically incorrect, or meaningless.
14. Do NOT merge separate source segments in a way that causes information to disappear.
15. Every input segment must have corresponding translated content.
16. Do not add explanations, comments, corrections, or assumptions.
17. Do not use outside knowledge to repair or interpret the source content.
18. If the source contains an unclear/noisy phrase, translate whatever is understandable and preserve the uncertainty of the remaining part.

THE SOURCE CONTENT IS THE SOURCE OF TRUTH.

{source_label}RAW SOURCE CONTENT (Segment {idx}/{len(batches)}):
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

    final_translated_content = "\n".join(translated_batches).strip()

    if DEBUG:
        print("\n===== TRANSLATION BATCH METRICS =====")
        print(f"source_filename: {os.path.basename(source_filename)}")
        print(f"target_language: {target_lang_display}")
        print(f"translated_batch_count: {translated_batch_count}")
        print(f"input_char_length: {len(full_content)}")
        print(f"output_char_length: {len(final_translated_content)}")

    return final_translated_content, translated_batch_count


def ask(question, active_file=None, modality=None, return_structured=False):
    """
    Main Multimodal RAG Orchestrator function.
    Extracts QueryPlan and dispatches execution to IntentRouter.
    Supports Mode A (Modality Chat via modality param) and Mode B (Sidebar File Chat via active_file param).
    """
    from app.rag.intent_router import IntentRouter
    from app.services.face_service import register_pending_face
    from app.services.interaction_state import (
        get_pending_faces,
        clear_pending_faces
    )
    from app.query.query_plan import Modality, RequestScope
    from app.query.query_analyzer import resolve_canonical_source_id

    question_lower = question.lower().strip()

    # Step 1: Query Analysis & QueryPlan Construction
    analysis = analyze_query(question, active_file=active_file, request_modality=modality)
    plan = analysis.get("plan")

    # Override / Apply explicit active_file context (Mode B: Sidebar File-Scoped Chat)
    if active_file:
        canonical = resolve_canonical_source_id(active_file) or active_file
        basename = os.path.basename(active_file)
        analysis["source_hint"] = basename
        analysis["canonical_source_id"] = canonical
        analysis["request_scope"] = "SINGLE_FILE"
        if plan:
            plan.scope = RequestScope.QUESTION_ANSWER
            plan.source_spec.source_hint = basename
            plan.source_spec.canonical_path = canonical
            plan.source_spec.is_explicit = True
            plan.source_spec.is_resolved = True
            plan.source_spec.is_ambiguous = False

    # Override / Apply explicit modality context (Mode A: Modality-Based Chat)
    if modality and modality.lower() != "all":
        mod_clean = modality.lower().strip()
        analysis["modality"] = mod_clean
        if plan:
            try:
                plan.modality = Modality(mod_clean)
            except ValueError:
                pass

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

    # 5. RELEVANT CHUNKS EVIDENCE
    evidence_items = evidence
    if isinstance(arg2, dict) and not evidence_items:
        evidence_items = arg2.get("evidence") or arg2.get("chunks") or []

    if evidence_items:
        chunk_blocks = []
        for idx, item in enumerate(evidence_items, 1):
            if isinstance(item, dict):
                src_name = os.path.basename(item.get("source") or item.get("filename") or "Chunk")
                c_id = item.get("chunk_id") or idx
                content_text = item.get("text") or item.get("content") or str(item)
                chunk_blocks.append(f"[{idx}] Source: {src_name} (Chunk #{c_id})\n{content_text.strip()}")
            elif isinstance(item, str):
                chunk_blocks.append(f"[{idx}] {item.strip()}")

        if chunk_blocks:
            sections.append(f"{header}\nRELEVANT CHUNKS\n{header}\n" + "\n\n".join(chunk_blocks))

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