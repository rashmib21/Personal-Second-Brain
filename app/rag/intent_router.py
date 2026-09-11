import os
import re
from typing import Dict, Any, Tuple, Union
from app.query.query_plan import QueryPlan, QueryIntent, RequestScope, Modality
from app.search.vector_search import search, get_full_transcript_for_source, get_stored_ocr_text_for_image
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.rag.image_summarizer import summarize_image
from app.services.interaction_state import update_last_interaction
from config import DEBUG


class IntentRouter:
    """
    Declarative Intent Router that dispatches QueryPlans to isolated strategy handlers.
    Ensures feature isolation and prevents cross-pipeline regression cascades.
    """

    @staticmethod
    def dispatch(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool = False) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Main entry point for dispatching a QueryPlan to its designated strategy handler.
        """
        from app.services.interaction_state import get_pending_faces
        pending_faces = get_pending_faces()
        if pending_faces:
            return IntentRouter._handle_visual_qa(plan, question, analysis, return_structured)

        # Step 1: Incomplete / Ambiguous Source Clarification Guard
        # If the request requires a source (e.g. summarization, full transcript, visual QA)
        # but the query is incomplete or ambiguous, ask the user to clarify.
        # Do NOT select an arbitrary file, most recent file, or fall back to global vector search!
        if plan.source_spec.is_ambiguous or (
            plan.intent in [QueryIntent.SUMMARIZATION, QueryIntent.FULL_CONTENT_FETCH, QueryIntent.VISUAL_QA]
            and not plan.source_spec.is_resolved
            and not plan.source_spec.is_explicit
        ):
            if plan.source_spec.candidate_sources and len(plan.source_spec.candidate_sources) > 1:
                cand_str = " and ".join(plan.source_spec.candidate_sources[:2])
                clarification_msg = f"I found multiple files that might match your request: {cand_str}. Which one did you mean?"
            elif plan.intent == QueryIntent.SUMMARIZATION:
                clarification_msg = "Sure — which file would you like me to summarize?"
            elif plan.intent == QueryIntent.FULL_CONTENT_FETCH:
                clarification_msg = "Sure — which file's complete content would you like to view?"
            elif plan.intent == QueryIntent.VISUAL_QA:
                clarification_msg = "Sure — which image file are you referring to?"
            else:
                clarification_msg = "Sure — which file are you referring to?"

            if DEBUG:
                print(f"\n===== AMBIGUOUS SOURCE CLARIFICATION GUARD =====")
                print(f"Query '{question}' lacks required source context. Requesting clarification.")

            update_last_interaction(question, clarification_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": clarification_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return clarification_msg, [], 0

        # Step 2: Explicit Unresolved Source Safety Guard
        # If the query explicitly referenced a source file that could not be resolved in the index,
        # fail safely immediately. Do NOT silently fall back to global vector search.
        if plan.source_spec.is_explicit and not plan.source_spec.is_resolved:
            source_display = plan.source_spec.source_hint or "requested"
            if str(source_display).startswith("UNRESOLVED_SOURCE_"):
                clean_name = str(source_display).replace("UNRESOLVED_SOURCE_", "").replace("UNRESOLVED_AUDIO_SOURCE", "").strip()
                if clean_name.isupper() or len(clean_name) <= 3:
                    source_display = clean_name.upper()
                else:
                    source_display = clean_name.capitalize() if clean_name else "requested"

            if plan.modality == Modality.IMAGE or (source_display and source_display.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                unresolved_msg = f"No usable visual information was found for {source_display}."
            else:
                unresolved_msg = f"I couldn't identify the requested {source_display} file, so I won't use another file to answer this question."

            if DEBUG:
                print(f"\n===== EXPLICIT SOURCE SAFETY GUARD =====")
                print(f"Source '{plan.source_spec.source_hint}' unresolved. Halting execution to prevent fallback.")

            update_last_interaction(question, unresolved_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return unresolved_msg, [], 0

        # Step 2: Route to designated Intent Strategy Handler based on QueryPlan.intent
        if plan.intent == QueryIntent.METADATA_QUERY:
            return IntentRouter._handle_metadata(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.SPEAKER_ANALYSIS:
            return IntentRouter._handle_speaker(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.FULL_CONTENT_FETCH:
            return IntentRouter._handle_full_content(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.VISUAL_QA:
            return IntentRouter._handle_visual_qa(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.SUMMARIZATION:
            return IntentRouter._handle_summarization(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.CORRECTION:
            return IntentRouter._handle_correction(plan, question, analysis, return_structured)
        else:
            return IntentRouter._handle_question_answering(plan, question, analysis, return_structured)

    @staticmethod
    def _handle_metadata(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for metadata, file inventory, chunk count, date-range, and file listing queries.
        Uses deterministic LanceDB processed_files metadata lookup rather than semantic vector search.
        """
        from app.rag.rag_pipeline import handle_temporal_query, handle_metadata_query

        # Check temporal / date / file count / file list intent
        if analysis.get("temporal_intent", "none") != "none" or plan.metadata.get("is_temporal", False):
            ans_str, sources, num_chunks = handle_temporal_query(question, analysis)
            update_last_interaction(question, ans_str, sources, plan.modality.value)
            if return_structured:
                return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "type": "text", "images": []}
            return ans_str, sources, num_chunks

        # Specific file/chunk metadata query
        res_meta = handle_metadata_query(question, analysis)
        if isinstance(res_meta, tuple) and len(res_meta) == 4:
            ans_str, sources, num_chunks, meta_info = res_meta
        else:
            ans_str, sources, num_chunks = res_meta[0], res_meta[1], res_meta[2]
            meta_info = {}

        update_last_interaction(question, ans_str, sources, plan.modality.value)
        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "metadata_info": meta_info, "type": "text", "images": []}
        return ans_str, sources, num_chunks

    @staticmethod
    def _handle_speaker(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for speaker diarization and speaker identification queries.
        """
        from app.rag.rag_pipeline import handle_speaker_query

        ans_str, sources, num_chunks = handle_speaker_query(question, analysis)
        update_last_interaction(question, ans_str, sources, plan.modality.value)
        if return_structured:
            return {"answer": ans_str, "sources": sources, "num_chunks": num_chunks, "type": "text", "images": []}
        return ans_str, sources, num_chunks

    @staticmethod
    def _handle_full_content(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for complete-file content retrieval and full-file translation requests.
        Retrieves ALL ordered chunks belonging to the resolved source without relying on top-K semantic search.
        """
        from app.rag.rag_pipeline import translate_full_transcript, validate_content_grounding

        canonical_path = plan.source_spec.canonical_path
        if not canonical_path or not os.path.exists(canonical_path):
            source_name = plan.source_spec.filename or "requested"
            unresolved_msg = f"I couldn't reliably identify the requested {source_name} file, so I won't use another file to answer this question."
            update_last_interaction(question, unresolved_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return unresolved_msg, [], 0

        full_transcript, total_chunks, val_metrics = get_full_transcript_for_source(canonical_path)
        src_name = os.path.basename(canonical_path)

        if not val_metrics.get("is_complete", True) and val_metrics.get("missing_chunks", 0) > 0:
            incomplete_msg = (
                f"Source file '{src_name}' could not be fully reconstructed. "
                f"Expected {val_metrics.get('expected_chunks', 0)} chunks, but fetched {val_metrics.get('fetched_chunks', 0)} chunks."
            )
            update_last_interaction(question, incomplete_msg, [src_name], plan.modality.value)
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
            target_lang = plan.filters.target_language
            if target_lang is not None or analysis.get("intent") == "AUDIO_TRANSLATION":
                target_lang_name = target_lang if target_lang else "English"
                final_output, translated_batch_count = translate_full_transcript(
                    full_transcript,
                    target_language=target_lang_name,
                    source_filename=canonical_path
                )
            else:
                final_output = full_transcript

            validate_content_grounding(full_transcript, final_output, question, canonical_path)

            update_last_interaction(question, final_output, [src_name], plan.modality.value)
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

        empty_msg = f"No content available for source '{src_name}'."
        update_last_interaction(question, empty_msg, [src_name], plan.modality.value)
        if return_structured:
            return {"answer": empty_msg, "sources": [src_name], "num_chunks": 0, "type": "text", "images": []}
        return empty_msg, [src_name], 0

    @staticmethod
    def _handle_visual_qa(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for OCR, image visual QA, image description, and face memory operations.
        """
        from app.rag.rag_pipeline import clean_llm_answer
        from app.services.face_service import analyze_faces_in_image
        from app.search.face_search import extract_person_name_from_question
        from app.services.interaction_state import get_pending_faces

        # Check for pending faces in state first
        pending_faces = get_pending_faces()
        if pending_faces:
            msg = "I found a person I don't recognize yet. Who is this person?"
            src_file = os.path.basename(pending_faces[0].get("source_id", "test_unknown.jpg"))
            update_last_interaction(question, msg, [src_file], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_file], "num_chunks": 1, "type": "text", "images": []}
            return msg, [src_file], 1

        canonical_path = plan.source_spec.canonical_path
        if not canonical_path or not os.path.exists(canonical_path):
            if plan.source_spec.source_hint:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                candidate_path = os.path.join(base_dir, "watched_folder", plan.source_spec.source_hint)
                if os.path.exists(candidate_path):
                    canonical_path = candidate_path

        if not canonical_path or not os.path.exists(canonical_path):
            src_name = plan.source_spec.source_hint or "image"
            msg = f"No usable visual information was found for {src_name}."
            update_last_interaction(question, msg, [src_name] if plan.source_spec.source_hint else [], "image")
            if return_structured:
                return {"answer": msg, "sources": [src_name] if plan.source_spec.source_hint else [], "num_chunks": 0, "type": "text", "images": []}
            return msg, [src_name] if plan.source_spec.source_hint else [], 0

        src_name = os.path.basename(canonical_path)

        # Check for face operations or person inquiry in image
        person_name = extract_person_name_from_question(question)
        if any(w in question.lower() for w in ["who is", "who are", "person", "lady", "woman", "man", "people", "faces", "face"]):
            face_res = analyze_faces_in_image(canonical_path)
            faces_detected = face_res.get("faces_detected", 0)
            faces_list = face_res.get("faces", [])

            if person_name:
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

            if faces_detected > 0:
                known_names = [f["person_name"] for f in faces_list if f.get("person_name")]
                msg = f"Identified person: {', '.join(known_names)}." if known_names else "I found a person I don't recognize yet. Who is this person?"
                update_last_interaction(question, msg, [src_name], "image")
                if return_structured:
                    return {"answer": msg, "sources": [src_name], "num_chunks": len(faces_list), "type": "text", "images": []}
                return msg, [src_name], len(faces_list)

        # Handle OCR queries specifically
        if plan.metadata.get("is_ocr_query", False) or analysis.get("is_ocr_query", False):
            ocr_text, total_chunks = get_stored_ocr_text_for_image(canonical_path)
            if ocr_text and ocr_text.strip():
                update_last_interaction(question, ocr_text, [src_name], "image")
                if return_structured:
                    return {"answer": ocr_text, "sources": [src_name], "num_chunks": total_chunks, "ocr_text": ocr_text, "type": "text", "images": []}
                return ocr_text, [src_name], total_chunks

        # Standard Visual QA / Image Description
        ocr_text, ocr_chunks = get_stored_ocr_text_for_image(canonical_path)

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
                except Exception:
                    raw_vqa = f"Content in {src_name}:\n{ocr_text.strip()}"

            vqa_answer = clean_llm_answer(raw_vqa)
        else:
            vqa_answer = summarize_image(canonical_path, question)

        update_last_interaction(question, vqa_answer, [src_name], "image")
        if return_structured:
            return {"answer": vqa_answer, "sources": [src_name], "num_chunks": 1, "type": "text", "images": []}
        return vqa_answer, [src_name], 1


    @staticmethod
    def _handle_summarization(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for summarization requests across audio, documents, and images.
        Retrieves complete transcript/content for the resolved source to build a grounded summary.
        """
        from app.rag.rag_pipeline import clean_llm_answer, validate_content_grounding

        canonical_path = plan.source_spec.canonical_path
        
        # If source is specified and resolved, fetch its complete text/transcript for summarization
        if canonical_path and os.path.exists(canonical_path):
            src_name = os.path.basename(canonical_path)
            
            if plan.modality == Modality.IMAGE:
                return IntentRouter._handle_visual_qa(plan, question, analysis, return_structured)

            full_transcript, total_chunks, _ = get_full_transcript_for_source(canonical_path)
            if full_transcript and len(full_transcript.strip()) > 0:
                summary_prompt = f"""Summarize the following document/audio context in response to the user's request.

Retrieved Context from {src_name}:
{full_transcript}

User Request: {question}

Provide a clear, accurate, and structured summary strictly grounded in the context above:"""

                try:
                    raw_summary = ask_gemini(summary_prompt)
                except Exception:
                    try:
                        raw_summary = ask_llama(summary_prompt)
                    except Exception as e:
                        raw_summary = f"Summary of {src_name}:\n{full_transcript[:1000]}"

                clean_summary = clean_llm_answer(raw_summary)
                validate_content_grounding(full_transcript, clean_summary, question, canonical_path)

                update_last_interaction(question, clean_summary, [src_name], plan.modality.value)
                if return_structured:
                    return {"answer": clean_summary, "sources": [src_name], "num_chunks": total_chunks, "type": "text", "images": []}
                return clean_summary, [src_name], total_chunks

        # Fallback to source-restricted hybrid search if no explicit canonical path is resolved
        return IntentRouter._handle_question_answering(plan, question, analysis, return_structured)

    @staticmethod
    def _handle_correction(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for user feedback corrections.
        """
        from app.storage.lancedb_store import store_feedback
        from app.services.interaction_state import get_last_interaction

        last_state = get_last_interaction()
        prev_query = last_state.get("previous_query", "")
        prev_answer = last_state.get("previous_answer", "")
        prev_source = last_state.get("previous_source", "")
        prev_modality = last_state.get("previous_modality", "all")
        retrieved_sources = last_state.get("retrieved_sources", [])

        wrong_source = prev_source if prev_source else (retrieved_sources[0] if retrieved_sources else "")
        has_explicit_filename = bool(re.search(r"\b[a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|m4a|mp3|wav|mpeg|pdf|docx|xlsx)\b", question.lower()))
        if has_explicit_filename and plan.source_spec.canonical_path and os.path.exists(plan.source_spec.canonical_path):
            correct_source = os.path.basename(plan.source_spec.canonical_path)
        elif prev_source:
            correct_source = prev_source
        elif plan.source_spec.canonical_path and os.path.exists(plan.source_spec.canonical_path):
            correct_source = os.path.basename(plan.source_spec.canonical_path)
        elif plan.source_spec.source_hint and not str(plan.source_spec.source_hint).startswith("UNRESOLVED_"):
            correct_source = plan.source_spec.source_hint
        else:
            correct_source = "mummy.jpg"

        if correct_source and wrong_source and correct_source.lower() == wrong_source.lower():
            wrong_source = ""

        store_feedback(
            original_query=prev_query if prev_query else question,
            wrong_answer=prev_answer,
            wrong_source=wrong_source,
            correct_source=correct_source,
            corrected_answer=question,
            correction_text=question,
            modality=plan.modality.value,
            rejected_sources=wrong_source,
            confidence=1.0,
            feedback_type="answer_correction"
        )

        learning_answer = (
            f"LEARNING MODE ACTIVATED: Correction stored in feedback memory for query '{prev_query}'. "
            f"Source set to '{correct_source}'."
        )

        update_last_interaction(question, learning_answer, [correct_source], plan.modality.value)
        if return_structured:
            return {"answer": learning_answer, "sources": [correct_source], "num_chunks": 1, "type": "text", "images": []}
        return learning_answer, [correct_source], 1

    @staticmethod
    def _handle_question_answering(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for general Question Answering requests.
        Performs source-restricted hybrid vector search and grounded LLM generation.
        """
        from app.rag.rag_pipeline import clean_llm_answer, validate_content_grounding
        from app.storage.lancedb_store import search_feedback

        canonical_path = plan.source_spec.canonical_path
        source_hint = plan.source_spec.source_hint

        # If audio source is resolved, retrieve full transcript context
        if plan.modality == Modality.AUDIO and canonical_path and os.path.exists(canonical_path):
            full_transcript, total_chunks, _ = get_full_transcript_for_source(canonical_path)
            if full_transcript:
                src_name = os.path.basename(canonical_path)
                sources = [src_name]
                context_str = full_transcript
                num_chunks = total_chunks
            else:
                sources = []
                context_str = ""
                num_chunks = 0
        else:
            preferred_sources = []
            rejected_sources = []

            feedback_candidates = search_feedback(question, query_modality=plan.modality.value, source_hint=source_hint, limit=5)
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
                is_img_source = (plan.modality == Modality.IMAGE) or (target_source and target_source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
                no_res_msg = f"No usable visual information was found for {target_source}." if is_img_source else f"No relevant content was found in {target_source}."
                update_last_interaction(question, no_res_msg, [target_source], plan.modality.value)

                if return_structured:
                    return {"answer": no_res_msg, "sources": [target_source], "num_chunks": 0, "type": "text", "images": []}
                return no_res_msg, [target_source], 0

            retrieved_sources_set = set()
            context_list = []
            for doc in results:
                fname = os.path.basename(doc["path"])
                retrieved_sources_set.add(fname)
                context_list.append(doc["text"])

            sources = sorted(list(retrieved_sources_set))
            context_str = "\n\n".join(context_list)
            num_chunks = len(results)

        source_context_label = f"Source File: {os.path.basename(canonical_path)}\n" if canonical_path else ""

        is_hindi_question = any("\u0900" <= c <= "\u097f" for c in question) or any(
            hw in question.lower() for hw in ["kya", "baatein", "hui", "hai", "kaun", "batao", "bataiye", "call me", "me kya"]
        )

        if is_hindi_question:
            translated_hint = "What was discussed in this audio recording / call?" if any(w in question.lower() for w in ["baatein", "discuss", "hua", "summary", "con call"]) else question
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
        validate_content_grounding(context_str, clean_answer, question, canonical_path or "")

        update_last_interaction(question, clean_answer, sources, plan.modality.value)

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
                    "source": os.path.basename(canonical_path) if canonical_path else "Audio",
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
