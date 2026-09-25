"""
Smalltalk and Short-Query Short-Circuit Module.

Provides early detection for exact greetings/chitchat and incomplete single-word queries,
bypassing expensive vector search and database retrieval operations.
"""

import re
from typing import Optional, Set


# Exact greeting phrases that return a fixed friendly response
EXACT_GREETINGS: Set[str] = {
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
}

# Stopwords and greeting words stripped when evaluating query meaningfulness
STOP_AND_GREETING_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could",
    "ought", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "their", "its", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "upon", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "don", "should", "now",
    "hi", "hii", "hello", "hey", "thanks", "ok", "okay", "bye", "yes", "hmm"
}


def normalize_query_text(raw_query: str) -> str:
    """
    Normalizes query text by lowercasing and stripping leading/trailing whitespace and punctuation.
    """
    if not raw_query:
        return ""
    
    cleaned = raw_query.lower().strip()
    cleaned = cleaned.strip("!?....,;:\"'()[]{}")
    return cleaned


def check_smalltalk_or_short_query(raw_query: str) -> Optional[str]:
    """
    Evaluates whether a query is an exact greeting or an incomplete query.

    Returns:
        - Fixed reply string if matched (greeting or incomplete query).
        - None if query contains sufficient meaningful content for RAG search.
    """
    if not raw_query or not raw_query.strip():
        return "Thoda aur bata sakte ho aap kya jaanna chahte ho?"

    normalized_text = normalize_query_text(raw_query)

    # STEP 1: Check for exact short greetings / chitchat
    if normalized_text in EXACT_GREETINGS:
        return "Hi! Kaise madad karun?"

    # STEP 2: Extract alphabetic words and count meaningful words
    # Replace underscores with spaces so terms like hi_direct_hire split into individual words
    text_for_tokens = normalized_text.replace("_", " ")
    words_list = re.findall(r"\b[a-zA-Z]+\b", text_for_tokens)

    meaningful_words = []
    for word in words_list:
        if word not in STOP_AND_GREETING_WORDS:
            meaningful_words.append(word)

    # If fewer than 2 meaningful words remain, prompt for clarification
    if len(meaningful_words) < 2:
        return "Thoda aur bata sakte ho aap kya jaanna chahte ho?"

    # Query has 2 or more meaningful words -> proceed to normal RAG search
    return None
