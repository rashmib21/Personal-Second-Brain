import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.query.query_plan import QueryPlan, QueryIntent, RequestScope, Modality
from app.query.query_analyzer import analyze_query, build_query_plan
from app.rag.rag_pipeline import ask


# ==============================================================================
# PROBLEM 1: Semantically Similar Queries Treated Consistently
# ==============================================================================
def test_problem_1_semantically_similar_summary_queries():
    """
    Verifies that different phrasings asking for document summaries/coverage
    produce consistent SUMMARIZATION intent and SUMMARY scope.
    """
    indexed_files = ["SQLNotesForProfessionals.pdf", "Behari_lal_call.m4a", "dense2.webp"]

    phrasings = [
        "give me a summary of SQL notes",
        "can you summarize my SQL notes?",
        "I want an overview of my SQL notes",
        "what is covered in my SQL notes?",
        "walk me through my SQL notes",
        "give me an outline of my SQL notes"
    ]

    for q in phrasings:
        plan = build_query_plan(q, indexed_files=indexed_files)
        assert plan.intent == QueryIntent.SUMMARIZATION, f"Failed on '{q}': expected SUMMARIZATION, got {plan.intent}"
        assert plan.scope == RequestScope.SUMMARY, f"Failed on '{q}': expected SUMMARY, got {plan.scope}"
        assert plan.source_spec.source_hint == "SQLNotesForProfessionals.pdf", f"Failed on '{q}': source hint should be SQLNotesForProfessionals.pdf, got {plan.source_spec.source_hint}"


# ==============================================================================
# PROBLEM 2: Document Summary vs Targeted QA Separation
# ==============================================================================
def test_problem_2_summary_vs_targeted_qa_separation():
    """
    Verifies that document summary queries use SUMMARIZATION intent while
    topic-specific questions use QUESTION_ANSWERING intent (targeted QA).
    """
    indexed_files = ["SQLNotesForProfessionals.pdf"]

    summary_query = "give me a summary of the SQL notes"
    plan_summary = build_query_plan(summary_query, indexed_files=indexed_files)
    assert plan_summary.intent == QueryIntent.SUMMARIZATION
    assert plan_summary.scope == RequestScope.SUMMARY

    qa_query_1 = "find information about joins in SQL notes"
    plan_qa_1 = build_query_plan(qa_query_1, indexed_files=indexed_files)
    assert plan_qa_1.intent == QueryIntent.QUESTION_ANSWERING
    assert plan_qa_1.scope == RequestScope.QUESTION_ANSWER
    assert plan_qa_1.source_spec.source_hint == "SQLNotesForProfessionals.pdf"

    qa_query_2 = "what does the SQL notes say about joins?"
    plan_qa_2 = build_query_plan(qa_query_2, indexed_files=indexed_files)
    assert plan_qa_2.intent == QueryIntent.QUESTION_ANSWERING
    assert plan_qa_2.scope == RequestScope.QUESTION_ANSWER
    assert plan_qa_2.source_spec.source_hint == "SQLNotesForProfessionals.pdf"


# ==============================================================================
# PROBLEM 3: Complete / All Content Requests
# ==============================================================================
def test_problem_3_complete_content_requests():
    """
    Verifies that requests for all/complete content set FULL_CONTENT_FETCH intent
    and COMPLETE_FILE scope rather than defaulting to top-K chunk retrieval.
    """
    indexed_files = ["SQLNotesForProfessionals.pdf", "Accounts.m4a"]

    complete_queries = [
        "give me all chapters",
        "list all chapters",
        "give me the complete content",
        "give me everything from this file"
    ]

    for q in complete_queries:
        plan = build_query_plan(q, indexed_files=indexed_files)
        assert plan.intent == QueryIntent.FULL_CONTENT_FETCH, f"Failed on '{q}': expected FULL_CONTENT_FETCH, got {plan.intent}"
        assert plan.scope == RequestScope.COMPLETE_FILE, f"Failed on '{q}': expected COMPLETE_FILE, got {plan.scope}"


# ==============================================================================
# PROBLEM 4: Metadata & Inventory Questions
# ==============================================================================
def test_problem_4_metadata_inventory_queries():
    """
    Verifies that metadata and inventory questions (including synonyms like photos)
    are classified as METADATA_QUERY and METADATA_ONLY scope.
    """
    inventory_queries = [
        "how many photos do I have?",
        "list all photos",
        "show all audio files",
        "what files are in my folder?"
    ]

    for q in inventory_queries:
        plan = build_query_plan(q)
        assert plan.intent == QueryIntent.METADATA_QUERY, f"Failed on '{q}': expected METADATA_QUERY, got {plan.intent}"
        assert plan.scope == RequestScope.METADATA_ONLY, f"Failed on '{q}': expected METADATA_ONLY, got {plan.scope}"


# ==============================================================================
# PROBLEM 5: Explicit Source Failure Safety (Zero Random Fallback)
# ==============================================================================
def test_problem_5_explicit_source_failure_safety():
    """
    Verifies that an explicit reference to an unindexed file fails safely
    without falling back to global vector search across unrelated indexed files.
    """
    indexed_files = ["SQLNotesForProfessionals.pdf", "Behari_lal_call.m4a", "mummy.jpg"]

    query = "what does the physics notes say about gravity?"
    plan = build_query_plan(query, indexed_files=indexed_files)

    assert plan.source_spec.is_explicit == True, "Expected is_explicit to be True"
    assert plan.source_spec.is_resolved == False, "Expected is_resolved to be False"
    assert plan.source_spec.source_hint != "SQLNotesForProfessionals.pdf", "Should not match unrelated SQL file"

    ans, sources, num_chunks = ask(query)
    assert len(sources) == 0, f"Expected 0 sources, got {sources}"
    assert num_chunks == 0
    assert "couldn't identify" in ans.lower() or "unresolved" in ans.lower() or "could not find" in ans.lower()


# ==============================================================================
# PROBLEM 6: Natural Source References & Ambiguity Detection
# ==============================================================================
def test_problem_6_natural_source_references():
    """
    Verifies natural references ("SQL notes") resolve uniquely to canonical filenames,
    and ambiguous references (matching multiple files) request clarification.
    """
    # 1. Unique resolution
    indexed_unique = ["SQLNotesForProfessionals.pdf", "Accounts.m4a"]
    plan_unique = build_query_plan("give me a summary of my SQL notes", indexed_files=indexed_unique)
    assert plan_unique.source_spec.is_resolved == True
    assert plan_unique.source_spec.source_hint == "SQLNotesForProfessionals.pdf"

    # 2. Ambiguous resolution (multiple matching files)
    indexed_ambiguous = ["SQLNotesForProfessionals.pdf", "SQLNotesAdvanced.pdf"]
    plan_ambig = build_query_plan("give me a summary of my SQL notes", indexed_files=indexed_ambiguous)
    assert plan_ambig.source_spec.is_ambiguous == True or plan_ambig.source_spec.confidence < 0.70
    assert len(plan_ambig.source_spec.candidate_sources) >= 1 or plan_ambig.source_spec.is_ambiguous


if __name__ == "__main__":
    print("Running 6 Problems Acceptance Suite...")
    test_problem_1_semantically_similar_summary_queries()
    print("PASS: Problem 1")
    test_problem_2_summary_vs_targeted_qa_separation()
    print("PASS: Problem 2")
    test_problem_3_complete_content_requests()
    print("PASS: Problem 3")
    test_problem_4_metadata_inventory_queries()
    print("PASS: Problem 4")
    test_problem_5_explicit_source_failure_safety()
    print("PASS: Problem 5")
    test_problem_6_natural_source_references()
    print("PASS: Problem 6")
    print("ALL 6 PROBLEM ACCEPTANCE TESTS PASSED!")
