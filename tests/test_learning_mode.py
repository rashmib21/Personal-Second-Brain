import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.rag.rag_pipeline import ask


def run_all_tests():
    print("=" * 80)
    print("RUNNING MULTIMODAL RAG & LEARNING MODE TEST SUITE")
    print("=" * 80)

    passed_count = 0
    total_tests = 9

    # -------------------------------------------------------------
    # TEST 1: Grounded Audio QA
    # -------------------------------------------------------------
    q1 = "What is the main topic discussed in the audio accounts?"
    print(f"\n[TEST 1] Query: '{q1}'")
    ans1, src1, n1 = ask(q1)
    print("  - Sources retrieved:", src1)
    print("  - Answer:\n", ans1)
    assert any("Accounts.m4a" in s for s in src1), "TEST 1 FAILED: Accounts.m4a was not retrieved!"
    passed_count += 1
    print("✓ TEST 1 PASSED")

    # -------------------------------------------------------------
    # TEST 2: Grounded Audio Summarization
    # -------------------------------------------------------------
    q2 = "can you summarize the account audio."
    print(f"\n[TEST 2] Query: '{q2}'")
    ans2, src2, n2 = ask(q2)
    print("  - Sources retrieved:", src2)
    print("  - Summary:\n", ans2)
    assert any("Accounts.m4a" in s for s in src2), "TEST 2 FAILED: Accounts.m4a was not retrieved!"
    # Ensure hallucinated details like purchases after midnight or account increased by 500 are absent
    forbidden_terms = ["purchases after midnight", "account increased by 500", "cash flow statement"]
    assert not any(term in ans2.lower() for term in forbidden_terms), "TEST 2 FAILED: Hallucinated terms present!"
    passed_count += 1
    print("✓ TEST 2 PASSED")

    # -------------------------------------------------------------
    # TEST 3: Speaker-Specific Query Unreliability
    # -------------------------------------------------------------
    q3 = "what does the speaker 2 discuss in account audio?"
    print(f"\n[TEST 3] Query: '{q3}'")
    ans3, src3, n3 = ask(q3)
    print("  - Sources retrieved:", src3)
    print("  - Answer:\n", ans3)
    assert any("Accounts.m4a" in s for s in src3), "TEST 3 FAILED: Accounts.m4a was not retrieved!"
    assert "cannot be determined" in ans3.lower(), "TEST 3 FAILED: Expected speaker unreliability response!"
    passed_count += 1
    print("✓ TEST 3 PASSED")

    # -------------------------------------------------------------
    # TEST 4: Source & Modality Routing (No Excel Retrieval for Audio Query)
    # -------------------------------------------------------------
    q4 = "is any girl speak in account audio?"
    print(f"\n[TEST 4] Query: '{q4}'")
    ans4, src4, n4 = ask(q4)
    print("  - Sources retrieved:", src4)
    print("  - Answer:\n", ans4)
    assert any("Accounts.m4a" in s for s in src4), "TEST 4 FAILED: Accounts.m4a was not retrieved!"
    assert not any("Germany_IT_Companies" in s for s in src4), "TEST 4 FAILED: Unrelated Excel file was wrongly retrieved!"
    assert "gender" in ans4.lower() or "cannot be determined" in ans4.lower(), "TEST 4 FAILED: Expected speaker gender unreliability response!"
    passed_count += 1
    print("✓ TEST 4 PASSED")

    # -------------------------------------------------------------
    # TEST 5: Correction Intent & Learning Mode Activation
    # -------------------------------------------------------------
    q5 = "you are incorrect the file source of above query is account audio."
    print(f"\n[TEST 5] Correction Query: '{q5}'")
    ans5, src5, n5 = ask(q5)
    print("  - Sources retrieved:", src5)
    print("  - Answer:\n", ans5)
    assert "learning mode activated" in ans5.lower(), "TEST 5 FAILED: Learning mode was not activated!"
    passed_count += 1
    print("✓ TEST 5 PASSED")

    # -------------------------------------------------------------
    # TEST 6: Future Query Benefiting from Learning Memory
    # -------------------------------------------------------------
    q6 = "is there a girl speaking in the account audio?"
    print(f"\n[TEST 6] Follow-up Query: '{q6}'")
    ans6, src6, n6 = ask(q6)
    print("  - Sources retrieved:", src6)
    print("  - Answer:\n", ans6)
    assert any("Accounts.m4a" in s for s in src6), "TEST 6 FAILED: Preferred Accounts.m4a source not retrieved!"
    assert not any("Germany_IT_Companies" in s for s in src6), "TEST 6 FAILED: Rejected source returned!"
    passed_count += 1
    print("✓ TEST 6 PASSED")

    # -------------------------------------------------------------
    # TEST 7: Negative Question - Exact Date
    # -------------------------------------------------------------
    q7 = "What was the exact date mentioned in the accounts audio?"
    print(f"\n[TEST 7] Negative Query: '{q7}'")
    ans7, src7, n7 = ask(q7)
    print("  - Sources retrieved:", src7)
    print("  - Answer:\n", ans7)
    assert "no exact date was mentioned" in ans7.lower(), "TEST 7 FAILED: Expected exact date negative answer!"
    passed_count += 1
    print("✓ TEST 7 PASSED")

    # -------------------------------------------------------------
    # TEST 8: Answer Correction Feedback
    # -------------------------------------------------------------
    q8 = "Your previous answer is wrong. The correct answer is that the speaker discusses accounting basics."
    print(f"\n[TEST 8] Answer Correction Query: '{q8}'")
    ans8, src8, n8 = ask(q8)
    print("  - Sources retrieved:", src8)
    print("  - Answer:\n", ans8)
    assert "learning mode activated" in ans8.lower(), "TEST 8 FAILED: Answer correction failed to trigger learning mode!"
    passed_count += 1
    print("✓ TEST 8 PASSED")

    # -------------------------------------------------------------
    # TEST 9: Structured 3-5 Point Audio Summary
    # -------------------------------------------------------------
    q9 = "Summarize the audio in 3-5 points"
    print(f"\n[TEST 9] Audio Summary Query: '{q9}'")
    ans9, src9, n9 = ask(q9)
    print("  - Sources retrieved:", src9)
    print("  - Answer:\n", ans9)
    assert len(src9) > 0, "TEST 9 FAILED: No audio sources retrieved!"
    assert any(s.endswith((".m4a", ".mp3", ".wav", ".mpeg")) for s in src9), "TEST 9 FAILED: Non-audio file retrieved!"
    passed_count += 1
    print("✓ TEST 9 PASSED")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_tests} TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
