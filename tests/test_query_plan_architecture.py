import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.query.query_plan import QueryPlan, QueryIntent, RequestScope, Modality
from app.query.query_analyzer import analyze_query, build_query_plan
from app.rag.intent_router import IntentRouter
from app.rag.rag_pipeline import ask


# ==============================================================================
# TEST 1: Semantic Equivalence (Different Phrasings -> Same QueryPlan)
# ==============================================================================
def test_semantic_equivalence_summarization_phrasings():
    """
    Verifies that semantically equivalent summarization queries produce identical QueryPlans.
    """
    indexed_files = ["Accounts.m4a", "Behari_lal_call.m4a", "dense2.webp", "mummy.jpg"]

    phrasings = [
        "summarize Behari_lal_call.m4a",
        "give me an overview of Behari_lal_call.m4a",
        "what was discussed in Behari_lal_call.m4a",
        "provide a brief synopsis of Behari_lal_call.m4a",
        "give me the main points of Behari_lal_call.m4a",
        "what are the key takeaways from Behari_lal_call.m4a"
    ]

    plans = [build_query_plan(q, indexed_files=indexed_files) for q in phrasings]

    for i, plan in enumerate(plans):
        assert plan.intent == QueryIntent.SUMMARIZATION, f"Failed on phrasing '{phrasings[i]}': expected SUMMARIZATION, got {plan.intent}"
        assert plan.scope == RequestScope.SUMMARY, f"Failed on phrasing '{phrasings[i]}': expected SUMMARY, got {plan.scope}"
        assert plan.modality == Modality.AUDIO, f"Failed on phrasing '{phrasings[i]}': expected AUDIO, got {plan.modality}"
        assert plan.source_spec.source_hint == "Behari_lal_call.m4a", f"Failed on phrasing '{phrasings[i]}': expected Behari_lal_call.m4a source"


# ==============================================================================
# TEST 2: Explicit Source Safety (Unresolved Source -> Zero Fallback)
# ==============================================================================
def test_explicit_source_safety_unresolved():
    """
    Verifies that explicit source requests with unresolved filenames fail safely
    without falling back to global vector search across unrelated files.
    """
    indexed_files = ["Accounts.m4a", "Behari_lal_call.m4a", "mummy.jpg"]

    query = "What is written in file non_existent_financial_report_2099.pdf?"
    plan = build_query_plan(query, indexed_files=indexed_files)

    assert plan.source_spec.is_explicit == True
    assert plan.source_spec.is_resolved == False

    ans, sources, num_chunks = ask(query)
    assert len(sources) == 0
    assert num_chunks == 0
    assert "couldn't reliably identify" in ans.lower() or "unresolved" in ans.lower()


# ==============================================================================
# TEST 3: Complete-File Content Retrieval vs Top-K Vector QA
# ==============================================================================
def test_complete_file_content_intent():
    """
    Verifies that complete file content requests set COMPLETE_FILE scope and FULL_CONTENT_FETCH intent.
    """
    indexed_files = ["Accounts.m4a", "Behari_lal_call.m4a"]

    phrasings = [
        "give me the full transcript of Accounts.m4a",
        "show the complete text of Accounts.m4a",
        "translate the entire transcript of Accounts.m4a into english",
        "convert everything in Accounts.m4a without summarizing"
    ]

    for q in phrasings:
        plan = build_query_plan(q, indexed_files=indexed_files)
        assert plan.intent == QueryIntent.FULL_CONTENT_FETCH, f"Failed on '{q}': expected FULL_CONTENT_FETCH, got {plan.intent}"
        assert plan.scope == RequestScope.COMPLETE_FILE, f"Failed on '{q}': expected COMPLETE_FILE, got {plan.scope}"


# ==============================================================================
# TEST 4: Metadata Inventory & Date-Based Queries
# ==============================================================================
def test_metadata_inventory_intent():
    """
    Verifies that file listing, count, and date queries are classified as METADATA_QUERY.
    """
    metadata_queries = [
        "how many audio files do I have?",
        "list all images added recently",
        "show files uploaded yesterday",
        "count of files in database"
    ]

    for q in metadata_queries:
        plan = build_query_plan(q)
        assert plan.intent == QueryIntent.METADATA_QUERY, f"Failed on '{q}': expected METADATA_QUERY, got {plan.intent}"
        assert plan.scope == RequestScope.METADATA_ONLY, f"Failed on '{q}': expected METADATA_ONLY, got {plan.scope}"


# ==============================================================================
# TEST 5: Feature Isolation (Intent Isolation)
# ==============================================================================
def test_intent_isolation_routing():
    """
    Verifies that different intents route to distinct, isolated strategy handlers without interference.
    """
    indexed_files = ["mummy.jpg", "Behari_lal_call.m4a"]

    qa_plan = build_query_plan("What color is the shirt in mummy.jpg?", indexed_files=indexed_files)
    vqa_plan = build_query_plan("Describe the image mummy.jpg", indexed_files=indexed_files)
    meta_plan = build_query_plan("How many images are named mummy?", indexed_files=indexed_files)

    assert qa_plan.intent != meta_plan.intent
    assert vqa_plan.intent == QueryIntent.VISUAL_QA or vqa_plan.intent == QueryIntent.SUMMARIZATION
    assert meta_plan.intent == QueryIntent.METADATA_QUERY


# ==============================================================================
# TEST 6: Ambiguous & Negative Queries
# ==============================================================================
def test_ambiguous_and_negative_queries():
    """
    Tests edge cases like general conversation, empty file references, and ambiguous queries.
    """
    plan1 = build_query_plan("Hello, how are you?")
    assert plan1.intent == QueryIntent.QUESTION_ANSWERING

    plan2 = build_query_plan("tell me something interesting")
    assert plan2.intent == QueryIntent.QUESTION_ANSWERING
