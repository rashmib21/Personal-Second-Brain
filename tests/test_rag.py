import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.rag.rag_pipeline import ask, format_response

def test_rag():
    print("=" * 70)
    print("TESTING RAG RETRIEVAL & SOURCE ATTRIBUTION")
    print("=" * 70)

    question = "What is the Personal Second Brain test document about?"
    answer, sources, num_chunks = ask(question)

    print("\nQuestion:", question)
    print("\nFormatted Response:\n")
    print(format_response(answer, sources, num_chunks))

if __name__ == "__main__":
    test_rag()
