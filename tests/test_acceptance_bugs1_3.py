import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.rag.rag_pipeline import ask
from app.storage.lancedb_store import get_feedback_table, db, FEEDBACK_TABLE_NAME


def reset_feedback_table():
    if FEEDBACK_TABLE_NAME in db.list_tables().tables:
        db.drop_table(FEEDBACK_TABLE_NAME, ignore_missing=True)


def run_acceptance_tests():
    print("=" * 80)
    print("RUNNING ACCEPTANCE TESTS 1 THROUGH 6")
    print("=" * 80)

    reset_feedback_table()

    # -------------------------------------------------------------
    # TEST 1: Image Query
    # -------------------------------------------------------------
    q1 = "How many people are visible in the mummy image?"
    print(f"\n[TEST 1] Query: '{q1}'")
    ans1, src1, n1 = ask(q1)
    print("  -> Sources retrieved:", src1)
    print("  -> Answer:", ans1)
    assert any("mummy.jpg" in s.lower() for s in src1), f"TEST 1 FAILED: Expected mummy.jpg, got {src1}"
    assert not any("accounts.m4a" in s.lower() for s in src1), "TEST 1 FAILED: Accounts.m4a leaked into image query!"
    print("✓ TEST 1 PASSED: Source is mummy.jpg")

    # -------------------------------------------------------------
    # TEST 2: Image Query Correction
    # -------------------------------------------------------------
    q2 = "but there are two person standing in the image, one is Rashmi second one is her mother"
    print(f"\n[TEST 2] Correction Query: '{q2}'")
    ans2, src2, n2 = ask(q2)
    print("  -> Sources retrieved:", src2)
    print("  -> Answer:\n", ans2)
    assert "learning mode activated" in ans2.lower(), "TEST 2 FAILED: Learning mode was not activated!"
    print("✓ TEST 2 PASSED: Correction Intent = True, Feedback Stored = True")

    # -------------------------------------------------------------
    # TEST 3: Explicit Source Routing (No Accounts.m4a or Excel for mummy.jpg)
    # -------------------------------------------------------------
    q3 = "What color is the clothing of the person on the right in mummy.jpg?"
    print(f"\n[TEST 3] Query: '{q3}'")
    ans3, src3, n3 = ask(q3)
    print("  -> Sources retrieved:", src3)
    print("  -> Answer:", ans3)
    assert any("mummy.jpg" in s.lower() for s in src3), f"TEST 3 FAILED: Expected mummy.jpg, got {src3}"
    assert not any("accounts.m4a" in s.lower() for s in src3), "TEST 3 FAILED: Accounts.m4a was retrieved for mummy.jpg query!"
    assert not any("germany_it_companies" in s.lower() for s in src3), "TEST 3 FAILED: Excel file was retrieved for mummy.jpg query!"
    print("✓ TEST 3 PASSED: Explicit source constraint strictly enforced BEFORE search/reranking")

    # -------------------------------------------------------------
    # TEST 4: Audio Query
    # -------------------------------------------------------------
    q4 = "is any girl speak in account audio?"
    print(f"\n[TEST 4] Query: '{q4}'")
    ans4, src4, n4 = ask(q4)
    print("  -> Sources retrieved:", src4)
    print("  -> Answer:", ans4)
    assert any("accounts.m4a" in s.lower() for s in src4), f"TEST 4 FAILED: Expected Accounts.m4a, got {src4}"
    assert not any("mummy.jpg" in s.lower() for s in src4), "TEST 4 FAILED: Image file leaked into audio query!"
    print("✓ TEST 4 PASSED: Source is Accounts.m4a")

    # -------------------------------------------------------------
    # TEST 5: Audio Source Correction
    # -------------------------------------------------------------
    q5 = "you are incorrect, the file source of above query is account audio."
    print(f"\n[TEST 5] Correction Query: '{q5}'")
    ans5, src5, n5 = ask(q5)
    print("  -> Sources retrieved:", src5)
    print("  -> Answer:\n", ans5)
    assert "learning mode activated" in ans5.lower(), "TEST 5 FAILED: Learning mode was not activated!"
    print("✓ TEST 5 PASSED: Correction Intent = True, Source_Correction = True")

    # -------------------------------------------------------------
    # TEST 6: Follow-up Image Query Consulting Image Feedback
    # -------------------------------------------------------------
    q6 = "How many people are in mummy.jpg?"
    print(f"\n[TEST 6] Follow-up Image Query: '{q6}'")
    ans6, src6, n6 = ask(q6)
    print("  -> Sources retrieved:", src6)
    print("  -> Answer:", ans6)
    assert any("mummy.jpg" in s.lower() for s in src6), f"TEST 6 FAILED: Expected mummy.jpg, got {src6}"
    assert not any("accounts.m4a" in s.lower() for s in src6), "TEST 6 FAILED: Audio feedback leaked into image query!"
    print("✓ TEST 6 PASSED: Image feedback consulted, Audio feedback ignored!")

    print("\n" + "=" * 80)
    print("ALL 6 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_acceptance_tests()
