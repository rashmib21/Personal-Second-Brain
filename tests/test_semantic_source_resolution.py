import pytest
import os
from app.query.query_analyzer import build_query_plan, analyze_query
from app.query.query_plan import QueryIntent, RequestScope, Modality
from app.rag.rag_pipeline import ask


def test_1_natural_audio_description_resolution():
    """
    Verifies that a natural description like 'the conference call' or 'management discussion audio'
    resolves to 'Behari_lal_call.m4a' even though 'conference' and 'management' are NOT in the filename stem.
    """
    indexed_files = ["Behari_lal_call.m4a", "Accounts.m4a", "quantum_notes.txt"]
    natural_queries = [
        "summarize the conference call",
        "tell me about the management discussion audio",
        "what was discussed in the call recording"
    ]

    for q in natural_queries:
        plan = build_query_plan(q, indexed_files=indexed_files)
        assert plan.source_spec.is_explicit or plan.source_spec.is_ambiguous or not plan.source_spec.is_resolved, f"Failed on '{q}'"
        if plan.source_spec.is_resolved:
            assert plan.source_spec.source_hint == "Behari_lal_call.m4a", f"Failed on '{q}': expected Behari_lal_call.m4a, got {plan.source_spec.source_hint}"


def test_2_natural_document_description_resolution():
    """
    Verifies that natural document descriptions like 'the database notes' resolve to SQL_notes.txt.
    """
    indexed_files = ["SQL_notes.txt", "quantum_notes.txt", "dense1.jpg"]
    q = "give me a summary of the database notes"

    plan = build_query_plan(q, indexed_files=indexed_files)
    assert plan.source_spec.is_explicit == True
    if plan.source_spec.is_resolved:
        assert plan.source_spec.source_hint == "SQL_notes.txt"


def test_3_conversational_anaphora_preservation():
    """
    Verifies that conversational context ('it', 'that recording') preserves previously established source
    and is NOT overridden by arbitrary database vector similarity.
    """
    from app.services.interaction_state import update_last_interaction, clear_last_interaction
    clear_last_interaction()

    indexed_files = ["Behari_lal_call.m4a", "Accounts.m4a"]

    update_last_interaction(
        query="Tell me about the conference call recording.",
        answer="The conference call recording discusses Q3 financial metrics.",
        sources=["Behari_lal_call.m4a"],
        modality="audio"
    )

    plan = build_query_plan("Summarize it.", indexed_files=indexed_files)
    assert plan.source_spec.is_resolved == True
    assert plan.source_spec.source_hint == "Behari_lal_call.m4a"


def test_4_unresolved_natural_description_safety():
    """
    Verifies that a natural reference targeting an unindexed topic ('astrophysics paper')
    is marked as explicit unresolved source and does NOT fall back to unrelated global retrieval.
    """
    indexed_files = ["quantum_notes.txt", "Behari_lal_call.m4a"]
    q = "summarize the astrophysics paper"

    plan = build_query_plan(q, indexed_files=indexed_files)
    assert plan.source_spec.is_explicit == True
    assert plan.source_spec.is_resolved == False
    assert plan.source_spec.source_hint != "quantum_notes.txt"

    ans, sources, num_chunks = ask(q)
    assert len(sources) == 0, f"Expected 0 sources, got {sources}"
    assert num_chunks == 0
