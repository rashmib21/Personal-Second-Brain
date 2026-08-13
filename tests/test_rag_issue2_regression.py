import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.search.vector_search import search

def run_issue2_regression_tests():
    print("=" * 80)
    print("RUNNING RAG RETRIEVAL & PRECISION REGRESSION TESTS (ALL 9 QUERIES)")
    print("=" * 80)

    # Query 1: Programming Languages
    q1 = "What programming languages are listed in the resume?"
    res1 = search(q1)
    print(f"\n[TEST 1] Query: '{q1}'")
    print(f"  - Relevant chunks: {len(res1)}")
    assert len(res1) > 0, "Test 1 Failed: 0 chunks returned"
    assert any("Java, Python, SQL" in doc["text"] for doc in res1), "Test 1 Failed: Missing Java, Python, SQL chunk"

    # Query 2: Databases
    q2 = "Which databases and database-related skills are mentioned?"
    res2 = search(q2)
    print(f"\n[TEST 2] Query: '{q2}'")
    print(f"  - Relevant chunks: {len(res2)}")
    assert len(res2) > 0, "Test 2 Failed: 0 chunks returned"
    assert any("MySQL" in doc["text"] and "Database Design" in doc["text"] for doc in res2), "Test 2 Failed: Missing MySQL chunk"

    # Query 3: Frontend
    q3 = "What frontend technologies are listed in the resume?"
    res3 = search(q3)
    print(f"\n[TEST 3] Query: '{q3}'")
    print(f"  - Relevant chunks: {len(res3)}")
    sources3 = [os.path.basename(doc["path"]) for doc in res3]
    print(f"  - Sources: {set(sources3)}")
    assert "linux.pdf" not in sources3, f"Test 3 Failed: linux.pdf incorrectly included in sources {sources3}"

    # Query 4: Internship
    q4 = "Where did Rashmi complete her Backend Development Internship?"
    res4 = search(q4)
    print(f"\n[TEST 4] Query: '{q4}'")
    print(f"  - Relevant chunks: {len(res4)}")
    assert len(res4) > 0, "Test 4 Failed: 0 chunks returned"
    assert any("TechRitzy" in doc["text"] for doc in res4), "Test 4 Failed: Missing TechRitzy internship chunk"

    # Query 5: GoHigh Rentals APIs
    q5 = "What APIs did Rashmi develop for GoHigh Rentals?"
    res5 = search(q5)
    print(f"\n[TEST 5] Query: '{q5}'")
    print(f"  - Relevant chunks: {len(res5)}")
    assert len(res5) > 0, "Test 5 Failed: 0 chunks returned"

    # Query 6: Stock Market API
    q6 = "Which API was used to obtain real-time stock market data?"
    res6 = search(q6)
    print(f"\n[TEST 6] Query: '{q6}'")
    print(f"  - Relevant chunks: {len(res6)}")
    assert any("Angel One" in doc["text"] for doc in res6), "Test 6 Failed: Missing Angel One Smart API chunk"

    # Query 7: Kafka Partitioning Key
    q7 = "What was used as the Kafka partitioning key?"
    res7 = search(q7)
    print(f"\n[TEST 7] Query: '{q7}'")
    print(f"  - Relevant chunks: {len(res7)}")
    assert len(res7) > 0, "Test 7 Failed: 0 chunks returned"
    assert any("stock symbols" in doc["text"].lower() and "kafka" in doc["text"].lower() for doc in res7), "Test 7 Failed: Missing stock symbols Kafka chunk"

    # Query 8: Summary
    q8 = "Summarize Rashmi's backend development experience using only the resume."
    res8 = search(q8)
    print(f"\n[TEST 8] Query: '{q8}'")
    print(f"  - Relevant chunks: {len(res8)}")
    assert len(res8) > 0, "Test 8 Failed: 0 chunks returned"

    # Query 9: Unrelated Salary
    q9 = "What is Rashmi's current salary?"
    res9 = search(q9)
    print(f"\n[TEST 9] Unrelated Query: '{q9}'")
    print(f"  - Relevant chunks: {len(res9)}")
    assert len(res9) == 0, f"Test 9 Failed: Expected 0 chunks for salary, got {len(res9)}"

    print("\n" + "=" * 80)
    print("ALL 9 REGRESSION TESTS PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    run_issue2_regression_tests()
