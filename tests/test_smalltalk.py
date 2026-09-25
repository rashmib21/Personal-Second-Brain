"""
Unit tests for Smalltalk and Short Query Short-Circuit logic (STEP 1 & STEP 2).
"""

import pytest
from app.query.smalltalk import check_smalltalk_or_short_query
from app.rag.rag_pipeline import ask


def test_step1_greeting_variants_match():
    """
    Test 15+ exact greeting/chitchat variants return fixed greeting reply
    and bypass vector/database retrieval.
    """
    greeting_variants = [
        "hi",
        "hii",
        "hello",
        "hey",
        "hii there",
        "good morning",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "bye",
        "yes",
        "no",
        "HI!",
        "Hello...",
        "  hey  ",
        "Good Morning!",
    ]

    for greeting in greeting_variants:
        reply = check_smalltalk_or_short_query(greeting)
        assert reply == "Hi! Kaise madad karun?", f"Failed to match greeting: '{greeting}'"

        # Verify full ask() pipeline return structure
        ans, sources, num_chunks = ask(greeting)
        assert ans == "Hi! Kaise madad karun?", f"ask() failed for greeting: '{greeting}'"
        assert sources == []
        assert num_chunks == 0


def test_step1_long_greetings_and_queries_do_not_match():
    """
    Test that queries with extra content (e.g. 'hi, can you tell me about backend roles')
    are NOT matched as exact greetings.
    """
    long_queries = [
        "hi, can you tell me about backend roles",
        "hi_direct_hire companies",
        "hello please search for my resume",
        "hey what is the salary in indore",
        "thanks for the information about data engineering",
    ]

    for q in long_queries:
        reply = check_smalltalk_or_short_query(q)
        assert reply is None, f"Query wrongly short-circuited as greeting/incomplete: '{q}'"


def test_step2_short_incomplete_queries_clarification():
    """
    Test that incomplete/meaningless queries (< 2 meaningful words) return clarification prompt.
    """
    short_queries = [
        "which",
        "hmm",
        "what",
        "the",
        "a",
        "is",
        "can you",
    ]

    for q in short_queries:
        reply = check_smalltalk_or_short_query(q)
        assert reply == "Thoda aur bata sakte ho aap kya jaanna chahte ho?", f"Failed clarification for: '{q}'"

        # Verify full ask() pipeline return structure
        ans, sources, num_chunks = ask(q)
        assert ans == "Thoda aur bata sakte ho aap kya jaanna chahte ho?", f"ask() failed for: '{q}'"
        assert sources == []
        assert num_chunks == 0


def test_step2_normal_queries_proceed_to_rag():
    """
    Test that valid queries with >= 2 meaningful words (e.g. 'backend roles')
    return None and proceed to normal search flow.
    """
    valid_queries = [
        "backend roles",
        "data science jobs",
        "companies in Indore",
        "python developer",
    ]

    for q in valid_queries:
        reply = check_smalltalk_or_short_query(q)
        assert reply is None, f"Valid query wrongly short-circuited: '{q}'"
