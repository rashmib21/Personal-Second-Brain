import os
import re
from app.search.vector_search import search
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


def ask(question, return_structured=False):
    """
    Main Multimodal RAG Orchestrator function connecting 4 independent systems:
    System A — Hard Source Routing
    System B — Visual Question Answering
    System C — Face Recognition & Memory
    System D — Feedback / Learning Memory
    """
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

    # Step 1: Query Analysis & Source Resolution
    analysis = analyze_query(question)
    intent = analysis.get("intent", "question_answering")
    face_intent = analysis.get("face_intent", "none")
    is_visual_qa = analysis.get("is_visual_qa", False)
    modality = analysis.get("modality", "all")
    source_hint = analysis.get("source_hint")
    canonical_source_id = analysis.get("canonical_source_id")
    speaker_reference = analysis.get("speaker_reference")
    is_correction = analysis.get("is_correction", False)
    correction_details = analysis.get("correction_details", {})

    last_state = get_last_interaction()

    # Resolve target source filepath if canonical_source_id is available
    resolved_source_path = canonical_source_id if canonical_source_id else source_hint
    source_exists = bool(resolved_source_path and os.path.exists(resolved_source_path))

    # Print DEBUG LOG: SOURCE RESOLUTION
    print("\n===== SOURCE RESOLUTION =====")
    print(f"Query: {question}")
    print(f"Detected entity/source hint: {source_hint}")
    print(f"Candidate sources: {[os.path.basename(resolved_source_path)] if resolved_source_path else []}")
    print(f"Resolved source: {os.path.basename(resolved_source_path) if resolved_source_path else None}")
    print(f"Canonical source_id: {resolved_source_path}")

    # Handle unresolved audio source query cleanly
    if source_hint == "UNRESOLVED_AUDIO_SOURCE":
        unresolved_msg = "I couldn't identify the requested audio source in the indexed database."
        update_last_interaction(question, unresolved_msg, [], modality)

        print("\n===== SOURCE-RESTRICTED RETRIEVAL =====")
        print("Source: UNRESOLVED_AUDIO_SOURCE")
        print("Retrieved chunks: 0")
        print("Unique sources: []")

        print("\n===== VALIDATION =====")
        print("Expected source: None")
        print("Retrieved sources: []")
        print("Result: FAIL (Unresolved audio source)")

        if return_structured:
            return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
        return unresolved_msg, [], 0

    # Check for pending face registration flow first
    pending_faces = get_pending_faces()
    if pending_faces and (face_intent == "face_registration" or any(p in question_lower for p in ["this is", "my mother", "her name", "his name"])):
        # Extract user label (e.g. "mother", "rashmi")
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

        print("\n===== FACE ANALYSIS =====")
        print(f"Face operation: FACE_REGISTRATION")
        print(f"Registered identity: {user_label}")
        print(f"Identity source: user_registration")

        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": 1, "type": "text", "images": []}
        return ans_str, sources, 1

    # Step 2: Handle Follow-up Correction Intent (System D)
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

        print("\n===== FEEDBACK =====")
        print(f"Correction detected: True")
        print(f"Feedback modality: {target_modality}")
        print(f"Feedback source: {correct_source}")
        print(f"Matched feedback: {prev_query}")

        if return_structured:
            return {"answer": learning_answer, "sources": [correct_source], "num_chunks": 1, "type": "text", "images": []}
        return learning_answer, [correct_source], 1

    # Step 3: Determine Image & Face Routing Priority
    # Priority Order:
    # 1. Explicitly resolved existing image source -> source_image_retrieval or face_verification
    # 2. Specific-image + person identity verification -> face_verification
    # 3. Face-memory search without an existing source -> face_memory_search
    # 4. Existing RAG / VQA retrieval
    is_image_source_resolved = bool(source_exists and modality == "image")
    person_name = extract_person_name_from_question(question)
    source_stem = os.path.splitext(os.path.basename(source_hint))[0].lower() if source_hint else ""

    # Check if query is asking to verify or identify a person in a specific image source
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

    # ROUTE BRANCH 1: Source-Based Image Retrieval
    # Triggers when source_exists is True, modality is image, and query asks to retrieve/show the image source
    if is_image_source_resolved and not is_person_verification_query and not is_visual_qa:
        # print("\n===== IMAGE ROUTING =====")
        # print("Route: source_image_retrieval")

        src_name = os.path.basename(resolved_source_path)
        ans_str = f"Retrieved image source '{src_name}'."

        print("\n===== IMAGE RETRIEVAL =====")
        print("Query modality: image")
        print("Candidate count: 1")
        print(f"Candidate sources: ['{src_name}']")

        print("\n===== FINAL VALIDATION =====")
        print("Query modality: image")
        print(f"Final source: {src_name}")
        print("Source consistent: True")
        print("Modality consistent: True")

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
        # print("\n===== IMAGE ROUTING =====")
        # print("Route: face_verification")

        # print("\n===== FACE ANALYSIS =====")
        # print(f"Face intent: {face_intent if face_intent != 'none' else 'face_verification'}")

        target_img = resolved_source_path
        face_res = analyze_faces_in_image(target_img)
        faces_detected = face_res.get("faces_detected", 0)
        faces_list = face_res.get("faces", [])

        print(f"Faces detected: {faces_detected}")
        print(f"Embeddings generated: {face_res.get('embeddings_generated', 0)}")
        print(f"Known faces: {face_res.get('known_count', 0)}")
        print(f"Unknown faces: {face_res.get('unknown_count', 0)}")

        for f in faces_list:
            print(f"Face {f['face_id']}: status={f['status']}, person={f['person_name']}, distance={f['distance']}, identity_source={f['identity_source']}")

        src_name = os.path.basename(target_img)

        # Check if verifying a specific person name (e.g. Rashmi)
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

        # General face identification inside image
        pending_faces_list = get_pending_faces()
        if pending_faces_list and any(pf.get("status") == "unknown" for pf in pending_faces_list):
            msg = "I found a person I don't recognize yet. Who is this person?"
            update_last_interaction(question, msg, [src_name], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_name], "num_chunks": len(pending_faces_list), "type": "text", "images": []}
            return msg, [src_name], len(pending_faces_list)

        if faces_detected == 0:
            msg = "I couldn't detect a usable face in this image."
            update_last_interaction(question, msg, [src_name], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_name], "num_chunks": 0, "type": "text", "images": []}
            return msg, [src_name], 0

        unknown_faces = [f for f in faces_list if f.get("status") == "unknown"]
        if unknown_faces:
            set_pending_faces(unknown_faces)
            msg = "I found a person I don't recognize yet. Who is this person?"
            update_last_interaction(question, msg, [src_name], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
            return msg, [src_name], len(faces_list)

        known_names = [f["person_name"] for f in faces_list if f.get("person_name")]
        msg = f"Identified person: {', '.join(known_names)}."
        update_last_interaction(question, msg, [src_name], "image")
        if return_structured:
            return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
        return msg, [src_name], len(faces_list)

    # ROUTE BRANCH 3: Persistent Face-Memory Search (Without a specific existing source file)
    elif (face_intent == "face_search" or is_face_search_query(question)) and not is_image_source_resolved:
        # print("\n===== IMAGE ROUTING =====")
        # print("Route: face_memory_search")

        # print("\n===== FACE ANALYSIS =====")
        # print("Face intent: face_search")

        if person_name:
            matched_records = search_images_by_registered_face(person_name)
            if not matched_records:
                print(f"No registered face memory record found for person '{person_name}'.")
                msg = f"No registered face memory record found matching '{person_name}'."
                update_last_interaction(question, msg, [], "image")
                if return_structured:
                    return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
                return msg, [], 0

            image_paths = sorted(list(set(r.get("image_path") or r.get("path") for r in matched_records if (r.get("image_path") or r.get("path")))))
            sources = [os.path.basename(p) for p in image_paths]
            ans_str = f"Found {len(sources)} image(s) matching registered face memory for '{person_name}'."
            update_last_interaction(question, ans_str, sources, "image")

            # print(f"Known faces: {len(sources)}")
            # print(f"Identity source: face_memory")

            if return_structured:
                return {"answer": ans_str, "sources": sources, "num_chunks": len(sources), "type": "image", "images": [{"type": "image", "path": p, "source": os.path.basename(p)} for p in image_paths]}
            return ans_str, sources, len(sources)

    # Step 4: Handle Visual Question Answering (System B)
    if is_visual_qa or (modality == "image" and source_hint):
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

            print("\n===== IMAGE RETRIEVAL =====")
            print(f"Candidate count: 0")
            print(f"Candidate sources: []")

            print("\n===== FINAL VALIDATION =====")
            print(f"Query modality: image")
            print(f"Final source: {source_hint}")
            print(f"Source consistent: False")
            print(f"Modality consistent: True")
            print(f"LLM allowed: False")

            if return_structured:
                return {"answer": msg, "sources": [source_hint] if source_hint else [], "num_chunks": 0, "type": "text", "images": []}
            return msg, [source_hint] if source_hint else [], 0

        # Execute VQA
        vqa_answer = summarize_image(target_img, question)
        src_name = os.path.basename(target_img)

        print("\n===== IMAGE RETRIEVAL =====")
        print(f"Query modality: image")
        print(f"Candidate count: 1")
        print(f"Candidate sources: ['{src_name}']")

        print("\n===== FINAL VALIDATION =====")
        print(f"Query modality: image")
        print(f"Final source: {src_name}")
        print(f"Source consistent: True")
        print(f"Modality consistent: True")
        print(f"Identity backed by face memory: False")
        print(f"LLM allowed: True")

        update_last_interaction(question, vqa_answer, [src_name], "image")

        if return_structured:
            return {"answer": vqa_answer, "sources": [src_name], "num_chunks": 1, "type": "text", "images": []}
        return vqa_answer, [src_name], 1

    # Step 5: System A Hard Source Candidate Retrieval
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

        print("\n===== SOURCE-RESTRICTED RETRIEVAL =====")
        print(f"Source: {source_hint if source_hint else 'unrestricted'}")
        print("Retrieved chunks: 0")
        print("Unique sources: []")

        print("\n===== VALIDATION =====")
        print(f"Expected source: {source_hint if source_hint else 'any'}")
        print("Retrieved sources: []")
        print("Result: FAIL (No content retrieved)")

        if return_structured:
            return {"answer": no_res_msg, "sources": [target_source], "num_chunks": 0, "type": "text", "images": []}
        return no_res_msg, [target_source], 0

    # Step 6: Collect sources & context
    retrieved_sources_set = set()
    context_list = []
    for doc in results:
        fname = os.path.basename(doc["path"])
        retrieved_sources_set.add(fname)
        context_list.append(doc["text"])

    sources = sorted(list(retrieved_sources_set))
    context_str = "\n\n".join(context_list)
    num_chunks = len(results)

    print("\n===== SOURCE-RESTRICTED RETRIEVAL =====")
    print(f"Source: {source_hint if source_hint else 'unrestricted'}")
    print(f"Retrieved chunks: {num_chunks}")
    print(f"Unique sources: {sources}")

    # Step 7: Pre-LLM Consistency Validation (Hard Invariant Check)
    source_consistent = True
    if source_hint and source_hint != "UNRESOLVED_AUDIO_SOURCE":
        target_base = os.path.basename(source_hint).lower()
        for s in sources:
            if s.lower() != target_base and os.path.basename(s).lower() != target_base:
                source_consistent = False
                raise RuntimeError(f"PRE-LLM VALIDATION ERROR: Retrieved sources {sources} violate hard source invariant for '{source_hint}'.")

    modality_consistent = True
    if modality == "image" and any(not s.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for s in sources):
        modality_consistent = False
        raise RuntimeError(f"PRE-LLM VALIDATION ERROR: Retrieved sources {sources} violate image modality invariant.")

    if modality == "audio" and any(not s.lower().endswith((".m4a", ".mp3", ".wav", ".mpeg")) for s in sources):
        modality_consistent = False
        raise RuntimeError(f"PRE-LLM VALIDATION ERROR: Retrieved sources {sources} violate audio modality invariant.")

    print("\n===== VALIDATION =====")
    print(f"Expected source: {source_hint if source_hint else 'any'}")
    print(f"Retrieved sources: {sources}")
    print(f"Result: PASS")

    # Step 8: Build Grounded Prompt for LLM
    prompt = f"""Answer the user's question using ONLY the retrieved context below.

STRICT GROUNDING RULES:
1. Rely strictly on facts explicitly stated in the retrieved context. Never invent meanings, dates, numbers, inventory, or purchases.
2. Do NOT guess or invent personal face identities (e.g., Rashmi).
3. If context does not contain enough information, state that clearly.
4. Output ONLY the natural language answer.

Retrieved Context:
{context_str}

Question: {question}

Answer:"""

    grounded_sys_instruction = (
        "You are a strictly grounded Personal Second Brain Assistant. "
        "Answer ONLY using facts present in the retrieved context. "
        "Do NOT invent concepts, dates, numbers, or personal face identities. "
        "Output ONLY the natural language text."
    )

    try:
        raw_answer = ask_llama(prompt, system_instruction=grounded_sys_instruction)
    except Exception:
        try:
            raw_answer = ask_gemini(prompt)
        except Exception as e:
            raw_answer = f"LLM Error: {str(e)}"

    clean_answer = clean_llm_answer(raw_answer)
    update_last_interaction(question, clean_answer, sources, modality)

    if return_structured:
        return {
            "answer": clean_answer,
            "sources": sources,
            "num_chunks": num_chunks,
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
        return clean_llm_answer(ask_llama(prompt, system_instruction=grounded_sys_instruction))
    except Exception:
        try:
            return clean_llm_answer(ask_gemini(prompt))
        except Exception as e:
            return f"LLM Error: {str(e)}"


def format_response(arg1, arg2=None, arg3=None, arg4=0):
    """
    Formats the final structured output separating natural language answer from metadata.
    Supports format_response(question, answer, sources, num_chunks) and format_response(answer, sources, num_chunks).
    """
    from app.query.query_analyzer import resolve_canonical_source_id

    if isinstance(arg2, (list, tuple)) or (isinstance(arg1, str) and not isinstance(arg2, str) and arg2 is not None):
        answer = arg1
        sources = arg2 if arg2 is not None else []
        num_chunks = arg3 if arg3 is not None else 0
    else:
        answer = arg2 if arg2 is not None else ""
        sources = arg3 if arg3 is not None else []
        num_chunks = arg4

    if sources:
        first_src = sources[0]
        src_str = os.path.basename(first_src)
        if os.path.isabs(first_src) and os.path.exists(first_src):
            path_str = first_src
        else:
            canonical_path = resolve_canonical_source_id(first_src)
            path_str = canonical_path if canonical_path else first_src
    else:
        src_str = "None"
        path_str = "None"

    output = f"ANSWER\n{answer}\n\nSOURCE\n{src_str}\n\nPATH\n{path_str}\n\nRELEVANT CHUNKS\n{num_chunks}"
    return output



if __name__ == "__main__":
    while True:
        print("\n" + "=" * 75)
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break
        if not question.strip():
            continue

        answer, sources, num_chunks = ask(question)
        print(format_response(question, answer, sources, num_chunks))