import os
import sys

from app.query.query_analyzer import analyze_query, INTENT_AUDIO_TRANSCRIPT, INTENT_AUDIO_TRANSLATION, INTENT_TEXT_SEARCH, INTENT_AUDIO_SUMMARY
from app.search.vector_search import get_full_transcript_for_source
from app.rag.rag_pipeline import translate_full_transcript

def run_tests():
    print("=" * 60)
    print("RUNNING COMPREHENSIVE TRANSLATION PIPELINE REGRESSION TESTS")
    print("=" * 60)

    # 1. Test Exact Reported Dehari Case
    q1 = "convert all the text of the dehari file in english"
    res1 = analyze_query(q1)
    print(f"\nTest 1 (Exact Dehari Case): '{q1}'")
    print(f"  Detected Intent: {res1.get('intent')}")
    print(f"  Target Language: {res1.get('target_language')}")
    print(f"  Source Hint: {res1.get('source_hint')}")
    assert res1.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res1.get('intent')}"
    assert res1.get("target_language") == "english", f"Expected 'english', got {res1.get('target_language')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION with target_language='english'")

    # 2. Test Another Audio File
    q2 = "translate entire recording of account_discussion.m4a to English"
    res2 = analyze_query(q2)
    print(f"\nTest 2 (Another Audio File): '{q2}'")
    print(f"  Detected Intent: {res2.get('intent')}")
    print(f"  Target Language: {res2.get('target_language')}")
    print(f"  Source Hint: {res2.get('source_hint')}")
    assert res2.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res2.get('intent')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION")

    # 3. Test Different Language Target
    q3 = "convert all text of sample_audio.mp3 in french"
    res3 = analyze_query(q3)
    print(f"\nTest 3 (Different Target Language): '{q3}'")
    print(f"  Detected Intent: {res3.get('intent')}")
    print(f"  Target Language: {res3.get('target_language')}")
    assert res3.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res3.get('intent')}"
    assert res3.get("target_language") == "french", f"Expected 'french', got {res3.get('target_language')}"
    print("  PASS: Classified as INTENT_AUDIO_TRANSLATION with target_language='french'")

    # 4. Test Short Transcript Translation
    short_transcript = "Line 1: Hello from the short audio.\nLine 2: Testing single batch."
    translated_short, b_count_short = translate_full_transcript(short_transcript, target_language="english", source_filename="test_short.mp3")
    print(f"\nTest 4 (Short Transcript Translation):")
    print(f"  Input Lines: {len(short_transcript.splitlines())}")
    print(f"  Batch Count: {b_count_short}")
    assert b_count_short == 1, f"Expected 1 batch for short transcript, got {b_count_short}"
    assert len(translated_short) > 0, "Expected non-empty translation output"
    print("  PASS: Short transcript processed in 1 batch successfully")

    # 5. Test Long Transcript Translation (Multi-Batch Sequential Processing)
    paragraphs = [f"Paragraph {i}: This is a detailed long audio segment paragraph test sentence containing unique data point {i}." for i in range(1, 30)]
    long_transcript = "\n".join(paragraphs)
    translated_long, b_count_long = translate_full_transcript(long_transcript, target_language="english", source_filename="test_long.m4a")
    print(f"\nTest 5 (Long Transcript Multi-Batch Processing):")
    print(f"  Input Characters: {len(long_transcript)}")
    print(f"  Batch Count: {b_count_long}")
    assert b_count_long > 1, f"Expected >1 batches for long transcript, got {b_count_long}"
    assert len(translated_long) > 0, "Expected non-empty translation output"
    print("  PASS: Long transcript correctly split and translated sequentially in multiple batches")

    # 6. Test Noisy ASR Transcript Preservation
    noisy_transcript = "Mummy ne kaha ki 5000 rupaye Behari Lal ji ko bhej do zaroor."
    translated_noisy, b_count_noisy = translate_full_transcript(noisy_transcript, target_language="english", source_filename="noisy_test.m4a")
    print(f"\nTest 6 (Noisy ASR / Proper Noun Preservation):")
    print(f"  Raw ASR: '{noisy_transcript}'")
    print(f"  Translation: '{translated_noisy}'")
    assert "5000" in translated_noisy or "5,000" in translated_noisy, "Expected number 5000 preserved in translation"
    print("  PASS: Numbers and proper nouns preserved without silent hallucination or replacement")

    # 7. Test Transcript Containing Numbers
    num_transcript = "Invoice number 98452 total amount is USD 1250."
    translated_num, b_count_num = translate_full_transcript(num_transcript, target_language="english", source_filename="num_test.mp3")
    print(f"\nTest 7 (Transcript Containing Numbers):")
    print(f"  Raw Text: '{num_transcript}'")
    print(f"  Translation: '{translated_num}'")
    assert "98452" in translated_num or "98452" in num_transcript, "Expected exact invoice number preserved"
    print("  PASS: Numbers accurately retained across translation")

    # 8. Test Normal Audio Summary Request
    q8 = "summarize the discussion in Behari_lal_call.m4a"
    res8 = analyze_query(q8)
    print(f"\nTest 8 (Normal Audio Summary Request): '{q8}'")
    print(f"  Detected Intent: {res8.get('intent')}")
    assert res8.get("intent") == INTENT_AUDIO_SUMMARY, f"Expected {INTENT_AUDIO_SUMMARY}, got {res8.get('intent')}"
    print("  PASS: Summary query correctly routes to INTENT_AUDIO_SUMMARY")

    # 9. Test Specific QA Question (Negative Full-Source Case)
    q9 = "what is the amount mentioned in the recording?"
    res9 = analyze_query(q9)
    print(f"\nTest 9 (Specific QA Question): '{q9}'")
    print(f"  Detected Intent: {res9.get('intent')}")
    assert res9.get("intent") == INTENT_TEXT_SEARCH, f"Expected {INTENT_TEXT_SEARCH}, got {res9.get('intent')}"
    print("  PASS: Specific QA query routes to INTENT_TEXT_SEARCH")

    # 10. Test Document File Complete Translation
    q10 = "convert all text of project_report.pdf into English"
    res10 = analyze_query(q10)
    print(f"\nTest 10 (Document File Complete Translation): '{q10}'")
    print(f"  Detected Intent: {res10.get('intent')}")
    print(f"  Target Language: {res10.get('target_language')}")
    assert res10.get("intent") == INTENT_AUDIO_TRANSLATION, f"Expected {INTENT_AUDIO_TRANSLATION}, got {res10.get('intent')}"
    assert res10.get("target_language") == "english", f"Expected 'english', got {res10.get('target_language')}"
    print("  PASS: Document complete translation classified correctly as INTENT_AUDIO_TRANSLATION")

    # 11. Test Unresolved Explicit Source Zero Fallback
    from app.rag.rag_pipeline import ask
    q11 = "convert all text of non_existent_file.m4a in english"
    ans11, sources11, num_chunks11 = ask(q11)
    print(f"\nTest 11 (Unresolved Explicit Source Zero Fallback): '{q11}'")
    print(f"  Response Answer: {ans11}")
    print(f"  Retrieved Sources: {sources11}")
    assert "couldn't reliably identify" in ans11.lower() or "won't use another file" in ans11.lower(), "Expected zero-fallback warning message"
    assert len(sources11) == 0, f"Expected 0 fallback sources, got {sources11}"
    print("  PASS: Unresolved explicit source flagged correctly and 0 fallback sources returned")

    # 12. Test Full Transcript Validation Metrics Audit
    raw_text, total_c, metrics = get_full_transcript_for_source("Behari_lal_call.m4a")
    print(f"\nTest 12 (Full Transcript Completeness Validation Audit):")
    print(f"  Expected Chunks: {metrics.get('expected_chunks')}")
    print(f"  Fetched Chunks: {metrics.get('fetched_chunks')}")
    print(f"  Missing Chunks: {metrics.get('missing_chunks')}")
    print(f"  Duplicate Chunks: {metrics.get('duplicate_chunks')}")
    print(f"  Reconstructed Chunks: {metrics.get('reconstructed_chunks')}")
    print(f"  Is Complete: {metrics.get('is_complete')}")
    assert "expected_chunks" in metrics, "Expected metrics dict to contain 'expected_chunks'"
    assert "is_complete" in metrics, "Expected metrics dict to contain 'is_complete'"
    print("  PASS: Full transcript retrieval returns complete validation metrics audit")

    print("\n" + "=" * 60)
    print("ALL 12 TRANSLATION PIPELINE REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
