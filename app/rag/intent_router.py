import os
import re
import logging
from typing import Dict, Any, Tuple, Union
from app.query.query_plan import QueryPlan, QueryIntent, QueryOperation, RequestScope, Modality, FaceIntent
from app.search.vector_search import search, get_full_content_for_source, get_stored_ocr_text_for_image
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.rag.image_summarizer import summarize_image
from app.services.interaction_state import (
    update_last_interaction,
    get_pending_spreadsheet_query,
    set_pending_spreadsheet_query,
    clear_pending_spreadsheet_query,
)
from app.rag.spreadsheet_query import (
    load_spreadsheet_rows,
    filter_rows,
    sort_rows,
    group_by_city,
    format_rows,
)
from config import DEBUG
from app.query.query_analyzer import build_query_plan

logger = logging.getLogger(__name__)

@staticmethod
def _handle_source_type(plan, question, analysis, return_structured=False):
    import os

    canonical_path = plan.source_spec.canonical_path
    source_name = (
        plan.source_spec.filename
        or plan.source_spec.source_hint
        or os.path.basename(canonical_path or "")
    )

    if not canonical_path or not os.path.exists(canonical_path):
        answer = "The requested source could not be found."
    else:
        extension = os.path.splitext(canonical_path)[1].lower()

        image_extensions = {
            ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"
        }

        document_extensions = {
            ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf"
        }

        spreadsheet_extensions = {
            ".xls", ".xlsx", ".csv", ".ods"
        }

        if extension in image_extensions:
            file_type = "image"
        elif extension in document_extensions:
            file_type = "document"
        elif extension in spreadsheet_extensions:
            file_type = "spreadsheet"
        else:
            file_type = "file"

        answer = f"It is an {file_type}."

    if return_structured:
        return {
            "type": "text",
            "answer": answer,
            "sources": [source_name] if source_name else [],
        }

    return answer, [source_name] if source_name else [], 0

class IntentRouter:
    """
    Declarative Intent Router that dispatches QueryPlans to isolated strategy handlers.
    Ensures feature isolation and prevents cross-pipeline regression cascades.
    """

    @staticmethod
    def _handle_pending_spreadsheet_selection(
        question: str,
        return_structured: bool
    ):
        """
        Handles a filename selected after the router asked the user
        to choose between multiple spreadsheets.
        """

        pending = get_pending_spreadsheet_query()

        original_query = pending.get("query", "")
        candidates = pending.get("candidates", [])

        if not original_query or not candidates:
            return None

        user_filename = os.path.basename(
            question.strip().strip('"')
        ).lower()

        matches = [
            candidate
            for candidate in candidates
            if os.path.basename(candidate).lower() == user_filename
        ]

        # The current message is not an exact unique candidate.
        if len(matches) != 1:
            return None

        selected_path = matches[0]

        # Selection is complete.
        clear_pending_spreadsheet_query()

        # Rebuild the QueryPlan from the ORIGINAL query.
        original_plan = build_query_plan(original_query)

        # Force only the source selected by the user.
        original_plan.source_spec.canonical_path = selected_path
        original_plan.source_spec.source_hint = os.path.basename(
            selected_path
        )
        original_plan.source_spec.is_explicit = True
        original_plan.source_spec.is_resolved = True
        original_plan.source_spec.is_ambiguous = False
        original_plan.source_spec.confidence = 1.0
        original_plan.source_spec.candidate_sources = []

        logger.info(
            "Spreadsheet selection resolved: '%s' -> '%s'",
            original_query,
            selected_path
        )

        return IntentRouter._handle_spreadsheet_query(
            original_plan,
            original_query,
            original_plan.to_dict(),
            return_structured
        )


    def dispatch(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool = False) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Main entry point for dispatching a QueryPlan to its designated strategy handler.
        """
        # ---------------------------------------------------------
        # Pending spreadsheet clarification
        # ---------------------------------------------------------
        # If the previous turn asked the user to choose a spreadsheet,
        # treat the current filename as a source selection and resume
        # the ORIGINAL spreadsheet query.
        spreadsheet_selection_result = (
            IntentRouter._handle_pending_spreadsheet_selection(
                question,
                return_structured
            )
        )

        if spreadsheet_selection_result is not None:
            return spreadsheet_selection_result

        from app.services.interaction_state import get_pending_faces

        pending_faces = get_pending_faces()

        if (
            pending_faces
            and plan.intent == QueryIntent.IMAGE_FACE_QUERY
            and plan.face_intent == FaceIntent.IDENTIFICATION
        ):
            return IntentRouter._handle_visual_qa(
                plan,
                question,
                analysis,
                return_structured
            )

        # Step 1: Incomplete / Ambiguous Source Clarification Guard
        # If the request requires a single source file (e.g. full file summarization, full transcript fetch, visual QA)
        # but the query lacks source context and is NOT a topic search, ask the user to clarify.
        # Topic queries (e.g. "summarize what I wrote about data engineering") must proceed to broad retrieval.
        has_topic_context = any(
            phrase in question.lower()
            for phrase in ["about ", "regarding ", "on ", "what i ", "where i ", "my notes "]
        ) or analysis.get("request_scope") == "partial_topic"

        if plan.source_spec.is_ambiguous or (
            plan.intent in [QueryIntent.SUMMARIZATION, QueryIntent.FULL_CONTENT_FETCH, QueryIntent.VISUAL_QA]
            and not plan.source_spec.is_resolved
            and not plan.source_spec.is_explicit
            and not has_topic_context
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
            elif plan.intent == QueryIntent.SOURCE_TYPE:
                return IntentRouter._handle_source_type(
                    plan,
                    question,
                    analysis,
                    return_structured
                )    
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
                clean_name = str(source_display).replace("UNRESOLVED_SOURCE_", "").strip()               
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

        # Structured spreadsheet queries must bypass semantic RAG.
        if plan.intent == QueryIntent.SPREADSHEET_QUERY:
            return IntentRouter._handle_spreadsheet_query(
                plan,
                question,
                analysis,
                return_structured
            )

        elif plan.intent == QueryIntent.METADATA_QUERY:
            if plan.operation == QueryOperation.COUNT:
                return IntentRouter._handle_file_count(
                    plan,
                    question,
                    analysis,
                    return_structured
                )

            return IntentRouter._handle_metadata(
                plan,
                question,
                analysis,
                return_structured
            )
        
        elif plan.intent == QueryIntent.FULL_CONTENT_FETCH:
            return IntentRouter._handle_full_content(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.IMAGE_DISPLAY:
            # Dedicated image retrieval handler: returns actual image path to the frontend.
            # This handler must NOT re-examine the raw query for intent signals.
            return IntentRouter._handle_image_display(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.IMAGE_FACE_QUERY:
            # Face queries are classified by QueryAnalyzer and dispatched
            # according to the typed FaceIntent stored in QueryPlan.
            if plan.face_intent == FaceIntent.SEARCH:
                return IntentRouter._handle_face_search(
                    plan,
                    question,
                    analysis,
                    return_structured
                )

            return IntentRouter._handle_visual_qa(
                plan,
                question,
                analysis,
                return_structured
            )
        elif plan.intent == QueryIntent.VISUAL_QA:
            return IntentRouter._handle_visual_qa(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.SUMMARIZATION:
            return IntentRouter._handle_summarization(plan, question, analysis, return_structured)
        elif plan.intent == QueryIntent.CORRECTION:
            return IntentRouter._handle_correction(plan, question, analysis, return_structured)
        else:
            return IntentRouter._handle_question_answering(plan, question, analysis, return_structured)

    @staticmethod
    def _handle_spreadsheet_query(
        plan: QueryPlan,
        question: str,
        analysis: Dict[str, Any],
        return_structured: bool
    ) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Deterministic handler for structured spreadsheet queries.

        This bypasses vector search, BM25, RRF, CrossEncoder,
        and LLM generation. The complete spreadsheet is loaded
        and filtered/sorted directly.
        """

        canonical_path = None

        # ---------------------------------------------------------
        # 1. Use an already-resolved spreadsheet source if present
        # ---------------------------------------------------------
        if (
            plan.source_spec.canonical_path
            and os.path.exists(plan.source_spec.canonical_path)
        ):
            extension = os.path.splitext(
                plan.source_spec.canonical_path
            )[1].lower()

            if extension in {".xlsx", ".xls", ".csv", ".ods"}:
                canonical_path = plan.source_spec.canonical_path

        # ---------------------------------------------------------
        # 2. Find spreadsheets in watched_folder
        # ---------------------------------------------------------
        if not canonical_path:
            current_file = os.path.abspath(__file__)

            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(current_file)
                )
            )

            watched_folder = os.path.join(
                project_root,
                "watched_folder"
            )

            spreadsheet_extensions = {
                ".xlsx", ".xls", ".csv", ".ods"
            }

            candidates = []

            if os.path.isdir(watched_folder):
                for root, dirs, files in os.walk(watched_folder):
                    for filename in files:
                        extension = os.path.splitext(
                            filename
                        )[1].lower()

                        if extension in spreadsheet_extensions:
                            candidates.append(
                                os.path.join(root, filename)
                            )

            # One spreadsheet = unambiguous
            if len(candidates) == 1:
                canonical_path = candidates[0]

            # Multiple spreadsheets
            elif len(candidates) > 1:
                source_hint = None

                if plan.source_spec:
                    source_hint = (
                        getattr(
                            plan.source_spec,
                            "source_hint",
                            None
                        )
                        or getattr(
                            plan.source_spec,
                            "filename",
                            None
                        )
                    )

                if source_hint:
                    hint_name = os.path.basename(
                        source_hint
                    ).lower()

                    matching = [
                        p for p in candidates
                        if os.path.basename(p).lower()
                        == hint_name
                    ]

                    if len(matching) == 1:
                        canonical_path = matching[0]

                # If still ambiguous, don't randomly select a workbook.
                if not canonical_path:
                    names = [
                        os.path.basename(p)
                        for p in candidates[:5]
                    ]

                    msg = (
                        "I found multiple spreadsheet files. "
                        "Please specify which spreadsheet you want "
                        "to query: "
                        + ", ".join(names)
                    )

                    # Remember the original structured query so that
                    # the user's next filename reply can select the
                    # spreadsheet and continue the original request.
                    set_pending_spreadsheet_query(
                        question,
                        candidates
                    )

                    update_last_interaction(
                        question,
                        msg,
                        [],
                        plan.modality.value
                    )

                    if return_structured:
                        return {
                            "answer": msg,
                            "sources": [],
                            "num_chunks": 0,
                            "type": "text",
                            "images": []
                        }

                    return msg, [], 0

        # ---------------------------------------------------------
        # 3. No spreadsheet found
        # ---------------------------------------------------------
        if not canonical_path or not os.path.exists(canonical_path):
            msg = (
                "I couldn't find a spreadsheet file to use "
                "for this query."
            )

            update_last_interaction(
                question,
                msg,
                [],
                plan.modality.value
            )

            if return_structured:
                return {
                    "answer": msg,
                    "sources": [],
                    "num_chunks": 0,
                    "type": "text",
                    "images": []
                }

            return msg, [], 0

        # ---------------------------------------------------------
        # 4. Load & Execute Domain-Agnostic Structured Query
        # ---------------------------------------------------------
        from app.rag.spreadsheet_query import execute_spreadsheet_query

        try:
            answer, num_chunks = execute_spreadsheet_query(
                canonical_path,
                question
            )
        except Exception:
            logger.exception(
                "Failed to execute spreadsheet query on: %s",
                canonical_path
            )

            msg = f"I couldn't query the spreadsheet '{os.path.basename(canonical_path)}'."
            source_name = os.path.basename(canonical_path)

            update_last_interaction(
                question,
                msg,
                [source_name],
                plan.modality.value
            )

            if return_structured:
                return {
                    "answer": msg,
                    "sources": [source_name],
                    "num_chunks": 0,
                    "type": "text",
                    "images": []
                }

            return msg, [source_name], 0

        source_name = os.path.basename(canonical_path)

        # ---------------------------------------------------------
        # 5. Save interaction state & Return Result
        # ---------------------------------------------------------
        update_last_interaction(
            question,
            answer,
            [source_name],
            plan.modality.value
        )

        if return_structured:
            return {
                "answer": answer,
                "sources": [source_name],
                "num_chunks": num_chunks,
                "type": "text",
                "images": []
            }

        return answer, [source_name], num_chunks

    @staticmethod
    def _handle_metadata(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for metadata, file inventory, chunk count, date-range, and file listing queries.
        Uses deterministic LanceDB processed_files metadata lookup rather than semantic vector search.
        All filter parameters come from QueryPlan.filters — the authoritative source of truth.
        """
        from app.rag.rag_pipeline import handle_temporal_query, handle_metadata_query

        # Pass QueryPlan.filters fields directly so handle_temporal_query respects the typed plan.
        # The analysis dict is still passed as a legacy fallback for the intent/temporal_intent fields.
        ans_str, sources, num_chunks = handle_temporal_query(
            question,
            analysis,
            plan_modality=plan.modality.value,
            plan_start_dt=plan.filters.start_datetime,
            plan_end_dt=plan.filters.end_datetime,
            plan_date_label=plan.filters.date_label,
            plan_limit=plan.filters.extracted_limit,
        )
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
        from app.rag.rag_pipeline import translate_full_content, validate_content_grounding

        canonical_path = plan.source_spec.canonical_path
        if not canonical_path or not os.path.exists(canonical_path):
            source_name = plan.source_spec.filename or "requested"
            unresolved_msg = f"I couldn't reliably identify the requested {source_name} file, so I won't use another file to answer this question."
            update_last_interaction(question, unresolved_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return unresolved_msg, [], 0

        full_content, total_chunks, val_metrics = get_full_content_for_source(canonical_path)
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

        if full_content:
            target_lang = plan.filters.target_language
            if target_lang is not None:
                final_output, translated_batch_count = translate_full_content(
                    full_content,
                    target_language=target_lang,
                    source_filename=canonical_path
                )
            else:
                final_output = full_content

            validate_content_grounding(full_content, final_output, question, canonical_path)

            update_last_interaction(question, final_output, [src_name], plan.modality.value)
            if return_structured:
                return {
                    "answer": final_output,
                    "sources": [src_name],
                    "num_chunks": total_chunks,
                    "evidence": [{"source": src_name, "chunk_id": 1, "text": full_content}],
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
    def _handle_face_search(
        plan: QueryPlan,
        question: str,
        analysis: Dict[str, Any],
        return_structured: bool
    ) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handle cross-image face-memory search.

        QueryAnalyzer owns interpretation of the request.
        This handler consumes the typed FaceIntent and person name,
        then delegates matching to the existing persistent face-memory
        search implementation.
        """
        from app.search.face_search import search_images_by_face

        results = search_images_by_face(
            question,
            max_results=10
        )

        if not results:
            person_name = plan.face_person_name

            if person_name:
                msg = (
                    f"I couldn't find any images containing a stored "
                    f"face match for {person_name}."
                )
            else:
                msg = "I couldn't find any images containing faces."

            update_last_interaction(
                question,
                msg,
                [],
                "image"
            )

            if return_structured:
                return {
                    "answer": msg,
                    "sources": [],
                    "num_chunks": 0,
                    "type": "image",
                    "images": []
                }

            return msg, [], 0

        image_items = []
        sources = []

        for item in results:
            image_path = item.get("image_path") or item.get("path")

            if not image_path:
                continue

            source_name = item.get("source") or os.path.basename(image_path)

            image_items.append({
                "type": "image",
                "path": image_path,
                "source": source_name
            })

            sources.append(source_name)

        if not image_items:
            msg = (
                "I found face-memory matches, but no usable image "
                "files were available."
            )

            update_last_interaction(
                question,
                msg,
                [],
                "image"
            )

            if return_structured:
                return {
                    "answer": msg,
                    "sources": [],
                    "num_chunks": 0,
                    "type": "text",
                    "images": []
                }

            return msg, [], 0

        person_name = plan.face_person_name

        if person_name:
            msg = (
                f"I found {len(image_items)} image(s) containing "
                f"{person_name}."
            )
        else:
            msg = f"I found {len(image_items)} image(s) containing faces."

        update_last_interaction(
            question,
            msg,
            sources,
            "image"
        )

        if return_structured:
            return {
                "answer": msg,
                "sources": sources,
                "num_chunks": len(image_items),
                "type": "image",
                "images": image_items
            }

        return msg, sources, len(image_items)


    @staticmethod
    def _handle_visual_qa(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for OCR, image visual QA, image description, and face memory operations.
        """
        from app.rag.rag_pipeline import clean_llm_answer
        from app.services.face_service import analyze_faces_in_image
        from app.services.interaction_state import get_pending_faces

        # Check for pending faces in state first
        # Check pending face state ONLY for face-related requests.
        # Never allow pending face state to hijack unrelated queries.
        pending_faces = get_pending_faces()

        # QueryAnalyzer owns face-intent interpretation.
        # This handler only consumes the typed QueryPlan.
        if (
            pending_faces
            and plan.face_intent == FaceIntent.IDENTIFICATION
        ):
            msg = "I found a person I don't recognize yet. Who is this person?"
            src_file = os.path.basename(
                pending_faces[0].get("source_id", "test_unknown.jpg")
            )

            update_last_interaction(
                question,
                msg,
                [src_file],
                "image"
            )

            if return_structured:
                return {
                    "answer": msg,
                    "sources": [src_file],
                    "num_chunks": 1,
                    "type": "text",
                    "images": []
                }

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

        # ---------------------------------------------------------
        # Face-related visual questions
        # ---------------------------------------------------------
        # QueryAnalyzer has already classified the face intent.
        # Do NOT re-parse the natural-language question here.
        face_intent = plan.face_intent
        person_name = plan.face_person_name

        if face_intent != FaceIntent.NONE:
            face_res = analyze_faces_in_image(canonical_path)

            faces_detected = int(
                face_res.get("faces_detected", 0) or 0
            )
            faces_list = face_res.get("faces", []) or []

            # -----------------------------------------------------
            # 1. Explicit face/person COUNT query
            # -----------------------------------------------------
            if face_intent == FaceIntent.COUNT:
                if faces_detected == 0:
                    msg = f"No people were detected in {src_name}."
                elif faces_detected == 1:
                    msg = f"1 person is visible in {src_name}."
                else:
                    msg = f"{faces_detected} people are visible in {src_name}."

                update_last_interaction(
                    question,
                    msg,
                    [src_name],
                    "image"
                )

                if return_structured:
                    return {
                        "answer": msg,
                        "sources": [src_name],
                        "num_chunks": len(faces_list),
                        "type": "text",
                        "images": []
                    }

                return msg, [src_name], len(faces_list)

            # -----------------------------------------------------
            # 2. Explicit person identity query
            # -----------------------------------------------------
            if face_intent == FaceIntent.IDENTIFICATION:
                if person_name:
                    matching_face = None

                    for f in faces_list:
                        stored_name = f.get("person_name")

                        if (
                            stored_name
                            and stored_name.lower() == person_name.lower()
                            and f.get("status") == "known"
                        ):
                            matching_face = f
                            break

                    if matching_face:
                        msg = f"Yes, {person_name} was found in {src_name}."

                        update_last_interaction(
                            question,
                            msg,
                            [src_name],
                            "image"
                        )

                        if return_structured:
                            return {
                                "answer": msg,
                                "sources": [src_name],
                                "num_chunks": len(faces_list),
                                "type": "text",
                                "images": []
                            }

                        return msg, [src_name], len(faces_list)

                known_names = [
                    f.get("person_name")
                    for f in faces_list
                    if f.get("person_name")
                    and f.get("status") == "known"
                ]

                if known_names:
                    msg = f"Identified person(s): {', '.join(known_names)}."
                elif faces_detected > 0:
                    msg = (
                        f"I detected {faces_detected} face(s) in {src_name}, "
                        "but I don't have a stored identity for them."
                    )
                else:
                    msg = f"No faces were detected in {src_name}."

                update_last_interaction(
                    question,
                    msg,
                    [src_name],
                    "image"
                )

                if return_structured:
                    return {
                        "answer": msg,
                        "sources": [src_name],
                        "num_chunks": len(faces_list),
                        "type": "text",
                        "images": []
                    }

                return msg, [src_name], len(faces_list)

            # -----------------------------------------------------
            # 3. Explicit person-presence query
            # -----------------------------------------------------
            if face_intent == FaceIntent.PRESENCE:
                if faces_detected > 0:
                    msg = (
                        f"Yes. I detected {faces_detected} "
                        f"person/people in {src_name}."
                    )
                else:
                    msg = f"No. I did not detect any people in {src_name}."

                update_last_interaction(
                    question,
                    msg,
                    [src_name],
                    "image"
                )

                if return_structured:
                    return {
                        "answer": msg,
                        "sources": [src_name],
                        "num_chunks": len(faces_list),
                        "type": "text",
                        "images": []
                    }

                return msg, [src_name], len(faces_list)

            # SEARCH is intentionally not handled as a single-image
            # visual operation here. It requires cross-image face-memory
            # retrieval and will be routed separately.
        # Handle OCR queries specifically
        if plan.metadata.get("is_ocr_query", False) or analysis.get("is_ocr_query", False):
            ocr_text, total_chunks = get_stored_ocr_text_for_image(canonical_path)
            if ocr_text and ocr_text.strip():
                update_last_interaction(question, ocr_text, [src_name], "image")
                if return_structured:
                    return {"answer": ocr_text, "sources": [src_name], "num_chunks": total_chunks, "ocr_text": ocr_text, "type": "text", "images": []}
                return ocr_text, [src_name], total_chunks

        # ---------------------------------------------------------
        # Standard Visual QA / Image Description
        # ---------------------------------------------------------
        # IMPORTANT:
        # Image understanding must always use the actual image.
        # OCR is supporting evidence only and must never replace
        # VLM-based visual analysis.
        #
        # This prevents cases where tiny/garbled OCR text causes
        # an image-summary request to be answered from OCR alone.

        ocr_text, ocr_chunks = get_stored_ocr_text_for_image(canonical_path)

        try:
            vqa_answer = summarize_image(
                canonical_path,
                question
            )
        except Exception as e:
            logger.warning(
                f"VLM image analysis failed for {canonical_path}: {e}"
            )
            vqa_answer = ""

        # If VLM returned a useful answer, use it directly.
        if vqa_answer and vqa_answer.strip():
            vqa_answer = clean_llm_answer(vqa_answer)

        # Only use OCR as a fallback when VLM genuinely produced
        # no usable result.
        if not vqa_answer or not vqa_answer.strip():
            if ocr_text and ocr_text.strip():
                vqa_answer = (
                    f"Visual analysis was unavailable for '{src_name}'. "
                    f"The following text was extracted from the image:\n\n"
                    f"{ocr_text.strip()}"
                )
            else:
                vqa_answer = (
                    f"I couldn't obtain usable visual information from "
                    f"'{src_name}'."
                )
        if return_structured:
            return {"answer": vqa_answer, "sources": [src_name], "num_chunks": 1, "type": "text", "images": []}
        return vqa_answer, [src_name], 1


    @staticmethod
    def _handle_image_display(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for image retrieval / display requests.
        Returns the image file path so the frontend renderer can display the actual image.
        Does NOT re-examine the raw query for intent signals; all decisions come from QueryPlan fields.
        """
        canonical_path = plan.source_spec.canonical_path

        # Attempt fallback resolution via watched_folder if canonical_path is missing
        if not canonical_path or not os.path.exists(canonical_path):
            if plan.source_spec.source_hint:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                candidate_path = os.path.join(base_dir, "watched_folder", plan.source_spec.source_hint)
                if os.path.exists(candidate_path):
                    canonical_path = candidate_path

        if not canonical_path or not os.path.exists(canonical_path):
            src_name = plan.source_spec.source_hint or "image"
            msg = f"No image file was found for '{src_name}'."
            update_last_interaction(question, msg, [], "image")
            if return_structured:
                return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return msg, [], 0

        src_name = os.path.basename(canonical_path)
        ans_str = f"Retrieved image: '{src_name}'."
        update_last_interaction(question, ans_str, [src_name], "image")

        if return_structured:
            return {
                "answer": ans_str,
                "sources": [src_name],
                "num_chunks": 1,
                "type": "image",
                "images": [{"type": "image", "path": canonical_path, "source": src_name}]
            }
        return ans_str, [src_name], 1

    @staticmethod
    def _handle_summarization(plan: QueryPlan, question: str, analysis: Dict[str, Any], return_structured: bool) -> Union[Tuple[str, list, int], Dict[str, Any]]:
        """
        Handler for summarization requests across documents and images.
        Retrieves complete transcript/content for the resolved source to build a grounded summary.
        """
        from app.rag.rag_pipeline import clean_llm_answer, validate_content_grounding

        canonical_path = plan.source_spec.canonical_path

        # Safety guard: if the user explicitly named a source that could not be resolved,
        # do NOT fall through to global vector search — that would silently answer from a
        # different file, which is a hallucination risk.
        if plan.source_spec.is_explicit and not plan.source_spec.is_resolved:
            source_display = plan.source_spec.source_hint or "requested"
            if str(source_display).startswith("UNRESOLVED_"):
                clean_name = str(source_display).replace("UNRESOLVED_SOURCE_", "").strip()                
                source_display = clean_name.capitalize() if clean_name else "requested"
            unresolved_msg = f"I couldn't identify the '{source_display}' file, so I won't summarize a different file to answer this question."
            update_last_interaction(question, unresolved_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": unresolved_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return unresolved_msg, [], 0

        # If source is specified and resolved, fetch its complete source content for summarization
        if canonical_path and os.path.exists(canonical_path):
            src_name = os.path.basename(canonical_path)

            if plan.modality == Modality.IMAGE:
                return IntentRouter._handle_visual_qa(plan, question, analysis, return_structured)

            full_content, total_chunks, _ = get_full_content_for_source(canonical_path)
            if full_content and len(full_content.strip()) > 0:
                summary_prompt = f"""Summarize the following document context in response to the user's request.
Retrieved Context from {src_name}:
{full_content}

User Request: {question}

Provide a clear, accurate, and structured summary strictly grounded in the context above:"""

                try:
                    raw_summary = ask_gemini(summary_prompt)
                except Exception:
                    try:
                        raw_summary = ask_llama(summary_prompt)
                    except Exception as e:
                        raw_summary = f"Summary of {src_name}:\n{full_content[:1000]}"

                clean_summary = clean_llm_answer(raw_summary)
                validate_content_grounding(full_content, clean_summary, question, canonical_path)

                update_last_interaction(question, clean_summary, [src_name], plan.modality.value)
                if return_structured:
                    return {"answer": clean_summary, "sources": [src_name], "num_chunks": total_chunks, "type": "text", "images": []}
                return clean_summary, [src_name], total_chunks

        # No explicit source or unresolved: perform source-restricted hybrid search for summary
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
        has_explicit_filename = bool(re.search(r"\b[a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp|gif|bmp|tiff?|pdf|doc|docx|txt|md|rtf|xls|xlsx|csv|ods|ppt|pptx)\b", question.lower()))
        if has_explicit_filename and plan.source_spec.canonical_path and os.path.exists(plan.source_spec.canonical_path):
            correct_source = os.path.basename(plan.source_spec.canonical_path)
        elif prev_source:
            correct_source = prev_source
        elif plan.source_spec.canonical_path and os.path.exists(plan.source_spec.canonical_path):
            correct_source = os.path.basename(plan.source_spec.canonical_path)
        elif plan.source_spec.source_hint and not str(plan.source_spec.source_hint).startswith("UNRESOLVED_"):
            correct_source = plan.source_spec.source_hint
        else:
            # No identifiable source — do not invent one.
            correct_source = ""

        if correct_source and wrong_source and correct_source.lower() == wrong_source.lower():
            wrong_source = ""

        # Do not store feedback unless a real correct source can be identified.
        # Never fall back to a hardcoded filename — that would poison feedback memory.
        if not correct_source:
            no_source_msg = "Correction noted, but I couldn't identify which file to associate it with. Please re-ask the question specifying the correct file."
            update_last_interaction(question, no_source_msg, [], plan.modality.value)
            if return_structured:
                return {"answer": no_source_msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
            return no_source_msg, [], 0

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
        Trusts QueryPlan.intent and QueryPlan.modality — does NOT re-classify intent
        using raw query keywords.
        """
        from app.rag.rag_pipeline import clean_llm_answer, validate_content_grounding
        from app.storage.lancedb_store import search_feedback

        canonical_path = plan.source_spec.canonical_path
        source_hint = plan.source_spec.source_hint

        # IMAGE_DISPLAY is handled in _handle_image_display; if somehow we reach here with
        # IMAGE intent and a resolved canonical_path, delegate correctly rather than
        # re-implementing inline logic.
        if plan.intent == QueryIntent.IMAGE_DISPLAY:
            return IntentRouter._handle_image_display(plan, question, analysis, return_structured)

        # Source-restricted hybrid retrieval
        # ---------------------------------------------------------
        # Unified retrieval for QA
        # ---------------------------------------------------------
        # IMPORTANT:
        # Do NOT send an entire source file to the LLM for        # ordinary QA. Source resolution and retrieval are separate:
        #
        #   resolved source -> hard filter -> hybrid retrieval
        #   -> reranking -> relevant chunks -> LLM
        #
        # Complete-file requests are handled separately by
        # _handle_full_content() / _handle_summarization().
        # ---------------------------------------------------------

        preferred_sources = []
        rejected_sources = []

        feedback_candidates = search_feedback(
            question,
            query_modality=plan.modality.value,
            source_hint=source_hint,
            limit=5
        )

        if feedback_candidates:
            for fb in feedback_candidates:
                correct_source = fb.get("correct_source")
                wrong_source = fb.get("wrong_source")

                if (
                    correct_source
                    and correct_source not in preferred_sources
                ):
                    # Feedback may influence ranking only when it is
                    # compatible with an explicitly resolved source_hint.
                    # Never force a single preferred source on open global queries when source_hint is None.
                    if (
                        source_hint
                        and (
                            correct_source.lower() == source_hint.lower()
                            or os.path.basename(correct_source).lower()
                            == os.path.basename(source_hint).lower()
                        )
                    ):
                        preferred_sources.append(correct_source)

                if (
                    wrong_source
                    and wrong_source not in rejected_sources
                    and source_hint
                ):
                    rejected_sources.append(wrong_source)

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
            target_source = source_hint or "the requested file"

            is_img_source = (
                plan.modality == Modality.IMAGE
                or (
                    target_source
                    and target_source.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    )
                )
            )

            no_res_msg = (
                f"No usable visual information was found for {target_source}."
                if is_img_source
                else f"No relevant content was found in {target_source}."
            )

            update_last_interaction(
                question,
                no_res_msg,
                [target_source],
                plan.modality.value
            )

            if return_structured:
                return {
                    "answer": no_res_msg,
                    "sources": [target_source],
                    "num_chunks": 0,
                    "type": "text",
                    "images": []
                }

            return no_res_msg, [target_source], 0

        # Build context ONLY from retrieved relevant chunks.
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
                    lang_instruction = (
                        f"The user asked in Hindi/Hinglish: '{question}'. "
                        "Provide a clear, helpful, and strictly grounded answer in natural Hindi or Hinglish "
                        "based ONLY on facts in the retrieved context."
                    )
        else:
            lang_instruction = "Respond in clear, professional English."
        prompt = (
            "Answer the user's question using ONLY the retrieved context below.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Rely strictly on facts explicitly stated in the retrieved context. Never invent meanings, dates, numbers, company names, family names, or personal names.\n"
            "2. Identify topics, people, and events directly from the text.\n"
            "3. If the user asks for names, aliases, or background of a person or entity, list ONLY the exact names explicitly stated in the context text.\n"
            "4. If the context contains unclear names or uncertain wording, preserve the retrieved wording or state that the information is unclear.\n"            f"5. {lang_instruction}\n\n"
            f"{source_context_label}Retrieved Context:\n"
            f"{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        grounded_sys_instruction = (
            "You are a strictly grounded Personal Second Brain Assistant. "
            "Answer ONLY using facts present in the retrieved context. "
            "Do NOT invent concepts, dates, numbers, company names, father names, or personal face identities."
        )

        try:
            raw_answer = ask_gemini(
                grounded_sys_instruction + "\n\n" + prompt
            )
        except Exception:
            try:
                raw_answer = ask_llama(
                    prompt,
                    system_instruction=grounded_sys_instruction
                )
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
            elif 'full_content' in locals() and full_content:
                evidence_list.append({
                    "source": os.path.basename(canonical_path) if canonical_path else "Unknown source",
                    "chunk_id": 1,
                    "text": full_content
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
    @staticmethod
    def _handle_file_count(plan, question, analysis, return_structured=False):
        import os

        watched_folder = os.path.abspath("watched_folder")

        if not os.path.isdir(watched_folder):
            answer = "The watched folder does not exist."

            if return_structured:
                return {
                    "type": "text",
                    "answer": answer,
                    "sources": [],
                }

            return answer, [], 0

        extension_groups = {
            "image": {
                ".jpg", ".jpeg", ".png", ".webp",
                ".gif", ".bmp", ".tiff", ".tif"
            },
            
            "pdf": {
                ".pdf"
            },
            "document": {
                ".doc", ".docx", ".txt",
                ".md", ".rtf"
            },
            "spreadsheet": {
                ".xls", ".xlsx", ".csv", ".ods"
            },
        }

        # QueryAnalyzer has already resolved the requested file type.
        # This handler only executes the COUNT operation.
        # file_type=None or file_type="all" both mean count every file,
        # routed to the else branch below.
        requested_type = getattr(plan, "file_type", None)
        if requested_type == "all":
            requested_type = None

        if requested_type and requested_type in extension_groups:
            extensions = extension_groups[requested_type]

            count = 0

            for root, dirs, files in os.walk(watched_folder):
                for filename in files:
                    extension = os.path.splitext(filename)[1].lower()

                    if extension in extensions:
                        count += 1

            if requested_type == "image":
                answer = f"There are {count} images in the folder."
            elif requested_type == "pdf":
                answer = f"There are {count} PDF files in the folder."
            elif requested_type == "document":
                answer = f"There are {count} documents in the folder."
            elif requested_type == "spreadsheet":
                answer = f"There are {count} spreadsheets in the folder."
            else:
                answer = f"There are {count} files in the folder."

        else:
            # Generic "how many files?"
            count = 0

            for root, dirs, files in os.walk(watched_folder):
                count += len(files)

            answer = f"There are {count} files in the folder."

        if return_structured:
            return {
                "type": "text",
                "answer": answer,
                "sources": [],
            }

        return answer, [], count
