import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.rag_pipeline import ask, format_response
from app.query.query_analyzer import analyze_query

def test_query(question):
    print("\n" + "=" * 80)
    print(f"QUERY: {question}")
    print("=" * 80)
    
    analysis = analyze_query(question)
    print(f"ANALYSIS: {analysis}")
    
    answer, sources, num_chunks = ask(question)
    print("\nANSWER:")
    print(answer)
    print(f"SOURCES: {sources}")
    print(f"CHUNKS: {num_chunks}")
    return answer, sources, num_chunks

def main():
    test_cases = [
        "summarize the audio of jethalal",
        "what are the name of jethalal?",
        "what are the name of jethlal 's friends?",
        "how many speakers are in the audio of jethlal?",
        "what are the names mentioned in the jethlal file?",
        "give me the names of dense images",
        "how many images I have by the names of dense?",
        "please list the name images which have named by dense.",
        "give me all the content of dense image.",
        "how many chunks you have for the file behari lal?",
        "who is speakers in behari lal file?",
        "how many person are talking in the audio behari lal?",
        "is there any lady talk in the behari lal?",
        "con call me kya baatein hui hai?"
    ]

    print("\n" + "#" * 80)
    print("RUNNING ALL ACCEPTANCE BUG TESTS")
    print("#" * 80)

    results = {}
    for q in test_cases:
        ans, srcs, chunks = test_query(q)
        results[q] = (ans, srcs, chunks)

    print("\n" + "#" * 80)
    print("SUMMARY OF ACCEPTANCE BUG TESTS")
    print("#" * 80)
    for q, (ans, srcs, chunks) in results.items():
        print(f"\nQ: {q}")
        print(f"Sources: {srcs}")
        print(f"Answer snippet: {str(ans)[:120]}...")

if __name__ == "__main__":
    main()
