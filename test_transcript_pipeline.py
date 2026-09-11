import os
import sys

from app.query.query_analyzer import analyze_query, INTENT_AUDIO_TRANSCRIPT, INTENT_AUDIO_TRANSLATION, INTENT_TEXT_SEARCH
from app.search.vector_search import get_full_transcript_for_source

def run_tests():
    print("=" * 60)
    print("RUNNING PIPELINE REGRESSION TESTS")
    print("=" * 60)

    # 1. Test Exact Reported Query
    q1 = "convert all the text of the dehari file in english"
    res1 = analyze_query(q1)
    print(f"\nTest 1 (Exact Case): '{q1}'")
    print(f"  Detected Intent: {res1.get('intent')}")
    print(f"  Target Language: {res1.get('target_language')}")
    print(f"  Source Hint: {res1.get('source_hint')}")
    assert res1.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res1.get('intent')}"
    assert res1.get("target_language") == "english", f"Expected 'english', got {res1.get('target_language')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION with target_language='english'")

    # 2. Test Unseen Variation 1
    q2 = "translate entire recording of account_discussion.m4a to English"
    res2 = analyze_query(q2)
    print(f"\nTest 2 (Unseen File & Phrasing): '{q2}'")
    print(f"  Detected Intent: {res2.get('intent')}")
    print(f"  Target Language: {res2.get('target_language')}")
    print(f"  Source Hint: {res2.get('source_hint')}")
    assert res2.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res2.get('intent')}"
    assert res2.get("target_language") == "english", f"Expected 'english', got {res2.get('target_language')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION with target_language='english'")

    # 3. Test Unseen Variation 2 (Full Transcript without Translation)
    q3 = "give me the complete content of lecture_notes.wav"
    res3 = analyze_query(q3)
    print(f"\nTest 3 (Full Content without Translation): '{q3}'")
    print(f"  Detected Intent: {res3.get('intent')}")
    print(f"  Target Language: {res3.get('target_language')}")
    print(f"  Source Hint: {res3.get('source_hint')}")
    assert res3.get("intent") == INTENT_AUDIO_TRANSCRIPT, f"Expected {INTENT_AUDIO_TRANSCRIPT}, got {res3.get('intent')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSCRIPT")

    # 4. Test Negative Case (Specific Question QA)
    q4 = "what is the total budget mentioned in Dehari?"
    res4 = analyze_query(q4)
    print(f"\nTest 4 (Negative Case - Specific QA Query): '{q4}'")
    print(f"  Detected Intent: {res4.get('intent')}")
    assert res4.get("intent") == INTENT_TEXT_SEARCH, f"Expected {INTENT_TEXT_SEARCH}, got {res4.get('intent')}"
    print("  PASS: Classified as INTENT_TEXT_SEARCH (does not trigger full-transcript pipeline)")

    # 5. Test Different Target Language
    q5 = "convert all the text of sample_audio.mp3 in spanish"
    res5 = analyze_query(q5)
    print(f"\nTest 5 (Different Target Language): '{q5}'")
    print(f"  Detected Intent: {res5.get('intent')}")
    print(f"  Target Language: {res5.get('target_language')}")
    assert res5.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res5.get('intent')}"
    assert res5.get("target_language") == "spanish", f"Expected 'spanish', got {res5.get('target_language')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION with target_language='spanish'")

    print("\n" + "=" * 60)
    print("ALL REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
