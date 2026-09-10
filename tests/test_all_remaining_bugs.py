import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.rag_pipeline import ask
from app.query.query_analyzer import analyze_query, find_best_matching_source, get_indexed_filenames


def test_all_14_queries():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE REGRESSION SUITE FOR ALL 14 QUERIES")
    print("=" * 80)

    # 1. Typo-Tolerant Source Resolution for Jethalal
    print("\n--- BUG #1 & BUG #7: JETHALAL TYPO RESOLUTION & AUDIO SUMMARY ---")
    q1 = "what are the name of jethlal 's friends?"
    analysis1 = analyze_query(q1)
    print(f"Query: '{q1}'")
    print(f"  -> Resolved source_hint: {analysis1.get('source_hint')}")
    print(f"  -> Confidence: {analysis1.get('source_confidence')}")
    assert analysis1.get('source_hint') and "Jethalal" in analysis1.get('source_hint'), f"FAILED: Jethalal not resolved for typo query! Got: {analysis1.get('source_hint')}"
    print("✓ PASSED: 'jethlal's' typo resolved to Jethalal audio file with high confidence!")

    q1_sum = "summarize the audio of jethalal"
    ans1_sum, src1_sum, _ = ask(q1_sum)
    print(f"\nQuery: '{q1_sum}'")
    print(f"  -> Sources: {src1_sum}")
    print(f"  -> Answer snippet: {ans1_sum[:150]}...")
    assert any("jethalal" in s.lower() for s in src1_sum), f"FAILED: Jethalal audio not retrieved! Got: {src1_sum}"
    print("✓ PASSED: Jethalal audio summary executed!")

    # 2. Cross-Source Fallback Prevention
    print("\n--- BUG #2: PREVENT CROSS-SOURCE FALLBACK FOR UNRESOLVED ENTITIES ---")
    q2 = "how many speakers are in the audio of jethlal?"
    ans2, src2, _ = ask(q2)
    print(f"Query: '{q2}'")
    print(f"  -> Sources: {src2}")
    print(f"  -> Answer: {ans2}")
    assert not any("accounts.m4a" in s.lower() for s in src2), "FAILED: Accounts.m4a leaked into Jethalal query!"
    print("✓ PASSED: Jethalal speaker query did not fall back to Accounts.m4a!")

    q2_names = "what are the names mentioned in the jethlal file?"
    ans2_names, src2_names, _ = ask(q2_names)
    print(f"\nQuery: '{q2_names}'")
    print(f"  -> Sources: {src2_names}")
    print(f"  -> Answer: {ans2_names}")
    assert not any("accounts.m4a" in s.lower() for s in src2_names), "FAILED: Accounts.m4a leaked into Jethalal names query!"
    assert "अंजू" not in ans2_names, "FAILED: Invented 'अंजू' from Accounts.m4a returned for Jethalal!"
    print("✓ PASSED: Jethalal names query did not retrieve Accounts.m4a or Anju!")

    # 3. Dense Image Filename Queries
    print("\n--- BUG #3: DENSE IMAGE FILENAME FILTER ---")
    q3_1 = "give me the names of dense images"
    ans3_1, src3_1, _ = ask(q3_1)
    print(f"Query: '{q3_1}'\n  -> Answer:\n{ans3_1}")
    assert "dense.jpeg" in ans3_1 and "dense2.webp" in ans3_1 and "dense3.webp" in ans3_1, "FAILED: dense image names missing!"
    assert "mummy.jpg" not in ans3_1 and "wallpaper" not in ans3_1.lower(), "FAILED: unrelated images returned for dense query!"

    q3_2 = "how many images I have by the names of dense?"
    ans3_2, src3_2, cnt3_2 = ask(q3_2)
    print(f"\nQuery: '{q3_2}'\n  -> Answer: {ans3_2}")
    assert "3" in str(ans3_2), f"FAILED: Expected count 3, got {ans3_2}"

    q3_3 = "please list the name images which have named by dense."
    ans3_3, src3_3, _ = ask(q3_3)
    print(f"\nQuery: '{q3_3}'\n  -> Answer:\n{ans3_3}")
    assert "dense.jpeg" in ans3_3 and "dense2.webp" in ans3_3 and "dense3.webp" in ans3_3, "FAILED: dense images missing for 'named by dense'!"
    assert len(src3_3) == 3, f"FAILED: Expected exactly 3 matching files, got {len(src3_3)} ({src3_3})"
    print("✓ PASSED: Dense image filename queries returned exactly dense.jpeg, dense2.webp, dense3.webp!")

    # 4. Image OCR Content Query
    print("\n--- BUG #4: IMAGE OCR CONTENT EXTRACTION ---")
    q4 = "give me all the content of dense image."
    ans4, src4, _ = ask(q4)
    print(f"Query: '{q4}'\n  -> Answer snippet:\n{ans4[:200]}...")
    assert "mitochondrion" not in ans4.lower(), "FAILED: invented biological 'mitochondrion' hallucination in OCR!"
    assert any(term in ans4 for term in ["Apical", "Golgi", "Nucleus", "Host", "GRA", "Dense"]), "FAILED: OCR text labels missing!"
    print("✓ PASSED: Dense image OCR content returned actual OCR labels without biological hallucinations!")

    # 5. Hinglish Audio Summary Intent
    print("\n--- BUG #5: HINGLISH AUDIO SUMMARY INTENT ---")
    q5 = "con call me kya baatein hui hai?"
    analysis5 = analyze_query(q5)
    print(f"Query: '{q5}'")
    print(f"  -> Intent: {analysis5.get('intent')}")
    print(f"  -> Source: {analysis5.get('source_hint')}")
    assert analysis5.get('intent') == "AUDIO_SUMMARY", f"FAILED: Expected AUDIO_SUMMARY intent, got {analysis5.get('intent')}"
    assert analysis5.get('source_hint') and "Behari" in analysis5.get('source_hint'), f"FAILED: Expected Behari lal audio source, got {analysis5.get('source_hint')}"

    ans5, src5, _ = ask(q5)
    print(f"  -> Answer snippet:\n{ans5[:150]}...")
    assert "माफ़ कीजिये, मैं समझ नहीं पाया" not in ans5, "FAILED: System failed to understand Hinglish query!"
    print("✓ PASSED: Hinglish audio discussion query correctly classified as AUDIO_SUMMARY!")

    # 6. Speaker Queries (No Speaker Diarization)
    print("\n--- SPEAKER QUERIES (TRANSPARENT LIMITATION) ---")
    speaker_queries = [
        "who is speakers in behari lal file?",
        "how many person are talking in the audio behari lal?",
        "is there any lady talk in the behari lal?"
    ]
    for sq in speaker_queries:
        ans_sq, _, _ = ask(sq)
        print(f"Query: '{sq}'\n  -> Answer: {ans_sq}")
        assert "diarization" in ans_sq.lower() or "can't reliably determine" in ans_sq.lower() or "cannot determine" in ans_sq.lower(), f"FAILED: Speaker query did not return transparent limitation for '{sq}'"
    print("✓ PASSED: Speaker queries returned transparent limitation without inventing speaker identities!")

    print("\n" + "=" * 80)
    print("ALL REMAINING BUGS REGRESSION SUITE PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_all_14_queries()
