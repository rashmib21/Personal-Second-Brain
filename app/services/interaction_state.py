"""
Interaction State Manager for Personal Second Brain RAG.
Maintains state of the previous interaction (query, answer, sources, modality)
to support correction queries like "you are incorrect the file source of above query is account audio".
"""

from datetime import datetime

# Global dictionary holding the last interaction state
_LAST_INTERACTION_STATE = {
    "previous_query": "",
    "previous_answer": "",
    "previous_source": "",
    "previous_modality": "",
    "last_image_source": "",
    "last_document_source": "",
    "last_spreadsheet_source": "",
    "retrieved_sources": [],
    "user_provided_identity_information": {},
    "pending_face_registration": [],
    "pending_spreadsheet_query": "",
    "pending_spreadsheet_candidates": [],
    "last_spreadsheet_context": {},
    "timestamp": ""
}


def update_last_interaction(query, answer, sources, modality="all", identity_info=None):
    """
    Updates the global interaction state after a RAG response is generated.
    Tracks overall last source as well as modality-specific active sources.
    """
    global _LAST_INTERACTION_STATE

    # Determine main source from list of sources if available
    main_source = ""
    if sources and len(sources) > 0:
        main_source = sources[0]

    # Store values in state dictionary
    _LAST_INTERACTION_STATE["previous_query"] = query
    _LAST_INTERACTION_STATE["previous_answer"] = answer
    _LAST_INTERACTION_STATE["previous_source"] = main_source
    _LAST_INTERACTION_STATE["previous_modality"] = modality
    _LAST_INTERACTION_STATE["retrieved_sources"] = sources
    if identity_info is not None:
        _LAST_INTERACTION_STATE["user_provided_identity_information"] = identity_info
    _LAST_INTERACTION_STATE["timestamp"] = str(datetime.now())

    # Update modality-specific active referents
    if main_source:
        src_lower = main_source.lower()
        if src_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")) or modality == "image":
            _LAST_INTERACTION_STATE["last_image_source"] = main_source
        if src_lower.endswith((".pdf", ".docx", ".doc", ".txt", ".md")) or modality in ["document", "pdf", "docx", "text"]:
            _LAST_INTERACTION_STATE["last_document_source"] = main_source
        if src_lower.endswith((".xlsx", ".xls", ".csv", ".ods")) or modality == "spreadsheet":
            _LAST_INTERACTION_STATE["last_spreadsheet_source"] = main_source


def get_last_interaction():
    """
    Returns a copy of the last interaction state.
    """
    global _LAST_INTERACTION_STATE
    return _LAST_INTERACTION_STATE.copy()


def set_pending_faces(pending_list):
    """
    Stores list of pending face dictionaries waiting for user registration.
    Each item contains: {source_id, face_id, bbox, embedding, created_at}
    """
    global _LAST_INTERACTION_STATE
    _LAST_INTERACTION_STATE["pending_face_registration"] = pending_list


def get_pending_faces():
    """
    Returns list of pending faces waiting for user registration.
    """
    global _LAST_INTERACTION_STATE
    return _LAST_INTERACTION_STATE.get("pending_face_registration", [])


def clear_pending_faces():
    """
    Clears pending face registration state.
    """
    global _LAST_INTERACTION_STATE
    _LAST_INTERACTION_STATE["pending_face_registration"] = []


def set_pending_spreadsheet_query(query, candidates):
    """
    Stores a structured spreadsheet query while waiting for the user
    to select one of multiple spreadsheet files.
    """
    global _LAST_INTERACTION_STATE

    _LAST_INTERACTION_STATE["pending_spreadsheet_query"] = query
    _LAST_INTERACTION_STATE["pending_spreadsheet_candidates"] = list(candidates)


def get_pending_spreadsheet_query():
    """
    Returns the pending spreadsheet query and candidate files.
    """
    global _LAST_INTERACTION_STATE

    return {
        "query": _LAST_INTERACTION_STATE.get(
            "pending_spreadsheet_query",
            ""
        ),
        "candidates": list(
            _LAST_INTERACTION_STATE.get(
                "pending_spreadsheet_candidates",
                []
            )
        )
    }


def clear_pending_spreadsheet_query():
    """
    Clears pending spreadsheet selection state.
    """
    global _LAST_INTERACTION_STATE

    _LAST_INTERACTION_STATE["pending_spreadsheet_query"] = ""
    _LAST_INTERACTION_STATE["pending_spreadsheet_candidates"] = []


def set_spreadsheet_context(context_dict):
    """
    Stores structured spreadsheet context from the last executed query.
    Contains: {target_file, target_sheet, entity_column, metric_column, filters, query_plan}
    """
    global _LAST_INTERACTION_STATE
    if isinstance(context_dict, dict):
        _LAST_INTERACTION_STATE["last_spreadsheet_context"] = context_dict.copy()


def get_spreadsheet_context():
    """
    Returns the structured spreadsheet context from the last executed query.
    """
    global _LAST_INTERACTION_STATE
    return _LAST_INTERACTION_STATE.get("last_spreadsheet_context", {}).copy()


def clear_last_interaction():
    """
    Resets the last interaction state (useful for testing or session reset).
    """
    global _LAST_INTERACTION_STATE
    _LAST_INTERACTION_STATE["previous_query"] = ""
    _LAST_INTERACTION_STATE["previous_answer"] = ""
    _LAST_INTERACTION_STATE["previous_source"] = ""
    _LAST_INTERACTION_STATE["previous_modality"] = ""
    _LAST_INTERACTION_STATE["last_image_source"] = ""
    _LAST_INTERACTION_STATE["last_document_source"] = ""
    _LAST_INTERACTION_STATE["last_spreadsheet_source"] = ""
    _LAST_INTERACTION_STATE["retrieved_sources"] = []
    _LAST_INTERACTION_STATE["user_provided_identity_information"] = {}
    _LAST_INTERACTION_STATE["pending_face_registration"] = []
    _LAST_INTERACTION_STATE["pending_spreadsheet_query"] = ""
    _LAST_INTERACTION_STATE["pending_spreadsheet_candidates"] = []
    _LAST_INTERACTION_STATE["last_spreadsheet_context"] = {}
    _LAST_INTERACTION_STATE["timestamp"] = ""



