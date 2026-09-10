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
    "retrieved_sources": [],
    "user_provided_identity_information": {},
    "pending_face_registration": [],
    "timestamp": ""
}


def update_last_interaction(query, answer, sources, modality="all", identity_info=None):
    """
    Updates the global interaction state after a RAG response is generated.
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


def clear_last_interaction():
    """
    Resets the last interaction state (useful for testing or session reset).
    """
    global _LAST_INTERACTION_STATE
    _LAST_INTERACTION_STATE["previous_query"] = ""
    _LAST_INTERACTION_STATE["previous_answer"] = ""
    _LAST_INTERACTION_STATE["previous_source"] = ""
    _LAST_INTERACTION_STATE["previous_modality"] = ""
    _LAST_INTERACTION_STATE["retrieved_sources"] = []
    _LAST_INTERACTION_STATE["user_provided_identity_information"] = {}
    _LAST_INTERACTION_STATE["pending_face_registration"] = []
    _LAST_INTERACTION_STATE["timestamp"] = ""

