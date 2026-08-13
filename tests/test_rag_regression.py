import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.search.vector_search import search

def test_rag_regression():
    print("=" * 80)
    print("RUNNING RAG RETRIEVAL REGRESSION TESTS (STEP 10 & 11)")
    print("=" * 80)

    # Test 1: Programming Languages
    q1 = "What programming languages are listed in the resume?"
    results1 = search(q1)
    print(f"\n[TEST 1] Query: '{q1}'")
    print(f"  - Relevant chunks returned: {len(results1)}")
    assert len(results1) > 0, "Test 1 Failed: 0 relevant chunks returned!"
    matched_content1 = any("Java, Python, SQL" in doc["text"] for doc in results1)
    print(f"  - Target content ('Java, Python, SQL') present: {matched_content1}")
    assert matched_content1, "Test 1 Failed: Target programming languages chunk not in results!"

    # Test 2: Databases
    q2 = "Which databases and database-related skills are mentioned?"
    results2 = search(q2)
    print(f"\n[TEST 2] Query: '{q2}'")
    print(f"  - Relevant chunks returned: {len(results2)}")
    assert len(results2) > 0, "Test 2 Failed: 0 relevant chunks returned!"
    matched_content2 = any("MySQL" in doc["text"] and "Database Design" in doc["text"] for doc in results2)
    print(f"  - Target content ('MySQL, Database Design') present: {matched_content2}")
    assert matched_content2, "Test 2 Failed: Target databases chunk not in results!"

    # Test 3: Backend Frameworks (Control case)
    q3 = "What backend frameworks does Rashmi mention?"
    results3 = search(q3)
    print(f"\n[TEST 3] Query: '{q3}'")
    print(f"  - Relevant chunks returned: {len(results3)}")
    assert len(results3) > 0, "Test 3 Failed: Control case failed!"
    matched_content3 = any("Spring Boot" in doc["text"] for doc in results3)
    print(f"  - Target content ('Spring Boot') present: {matched_content3}")
    assert matched_content3, "Test 3 Failed: Control framework chunk not in results!"

    # Test 4: Unrelated Query (Salary)
    q4 = "What is Rashmi's current salary?"
    results4 = search(q4)
    print(f"\n[TEST 4] Unrelated Query: '{q4}'")
    print(f"  - Relevant chunks returned: {len(results4)}")
    assert len(results4) == 0, f"Test 4 Failed: Expected 0 chunks for unrelated query, got {len(results4)}!"

    print("\n" + "=" * 80)
    print("ALL RAG REGRESSION TESTS PASSED 100%!")
    print("=" * 80)

if __name__ == "__main__":
    test_rag_regression()
