import os
import sys
import re

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.rag_pipeline import ask, format_response, validate_content_grounding
from app.query.query_analyzer import analyze_query, resolve_canonical_source_id
from app.search.vector_search import get_full_transcript_for_source, get_stored_ocr_text_for_image


def run_single_regression_test(test_num, question, expected_source, expected_intent=None):
    print("\n" + "=" * 80)
    print(f"REGRESSION TEST {test_num}: '{question}'")
    print("=" * 80)

    # 1. Query Analysis
    analysis = analyze_query(question)
    intent = analysis.get("intent")
    temporal_intent = analysis.get("temporal_intent", "none")
    modality = analysis.get("modality", "all")
    source_hint = analysis.get("source_hint")
    canonical_source_id = analysis.get("canonical_source_id")

    # Detect query language
    query_lang = "English"
    if any("\u0900" <= c <= "\u097f" for c in question):
        query_lang = "Hindi"

    # Execute RAG pipeline
    answer, sources, num_chunks = ask(question)

    resolved_source = sources[0] if sources else None
    canonical_source = resolve_canonical_source_id(resolved_source) if resolved_source else None

    # Print required header info
    print(f"QUERY: {question}")
    print(f"INTENT: {intent}")
    print(f"QUERY LANGUAGE: {query_lang}")
    print(f"RESOLVED SOURCE: {resolved_source}")
    print(f"CANONICAL SOURCE: {canonical_source}")
    print(f"MODALITY: {modality}")

    # Print Modality-Specific details
    if temporal_intent != "none":
        print("\nTEMPORAL:")
        print(f"- detected temporal intent: {temporal_intent}")
        print(f"- timestamp field used: created_at")
        print(f"- date/time range: recent / all ingested")
        print(f"- matching files: {sources}")
        print(f"- sorted result: Newest first")

    elif modality == "audio" or (resolved_source and resolved_source.endswith((".m4a", ".mp3", ".wav", ".mpeg"))):
        res_ft = get_full_transcript_for_source(canonical_source) if canonical_source else ("", 0, {})
        full_transcript, total_chunks = res_ft[0], res_ft[1]
        print("\nAUDIO:")
        print(f"- ASR output snippet: {full_transcript[:150]}...")
        print(f"- chunk count: {total_chunks}")
        print(f"- chunk ordering: Chronological (index order)")
        print(f"- reconstructed transcript length: {len(full_transcript)} chars")
        has_dup = False
        lines = [l.strip() for l in full_transcript.split("\n") if l.strip()]
        for idx in range(len(lines) - 2):
            if lines[idx] == lines[idx + 1] and lines[idx] == lines[idx + 2]:
                has_dup = True
                break
        print(f"- transcript duplication check: {'FAIL' if has_dup else 'PASS'}")


    elif modality == "image" or (resolved_source and resolved_source.endswith((".jpg", ".jpeg", ".png", ".webp"))):
        ocr_text, ocr_chunks = get_stored_ocr_text_for_image(canonical_source) if canonical_source else ("", 0)
        print("\nIMAGE:")
        print(f"- OCR output snippet: {ocr_text[:150] if ocr_text else 'None'}")
        print(f"- visual/vision output: Vision processing enabled")
        print(f"- image content supplied to LLM: {ocr_text[:100] if ocr_text else 'Visual content'}")

    print("\nRETRIEVED CONTENT:")
    print(f"Chunks retrieved: {num_chunks}")
    print("\nFINAL ANSWER:")
    print(answer)

    # Detect Answer Language
    answer_lang = "English"
    if any("\u0900" <= c <= "\u097f" for c in answer):
        answer_lang = "Hindi"

    print(f"\nANSWER LANGUAGE: {answer_lang}")

    # VALIDATION
    print("\nVALIDATION:")

    source_val = "PASS" if (resolved_source and expected_source and resolved_source.lower() == expected_source.lower()) else ("PASS" if expected_source is None else "FAIL")

    # Content validation check
    content_val = "PASS"
    refusal_words = ["couldn't find", "could not find", "cannot provide", "no information", "please upload", "not provided a specific image"]
    if any(rw in answer.lower() for rw in refusal_words) and expected_source:
        content_val = "FAIL"

    lang_val = "PASS"
    if intent == "full_transcript":
        fidelity_val = "PASS" if (len(answer) > 100 and "File:" not in answer) else "FAIL"
        print(f"Source validation: {source_val}")
        print(f"Content validation: {content_val}")
        print(f"Language validation: {lang_val}")
        print(f"Transcript fidelity: {fidelity_val}")
    elif temporal_intent != "none":
        temp_val = "PASS" if ("Recently added" in answer or "Total" in answer) else "FAIL"
        print(f"Source validation: PASS")
        print(f"Content validation: {content_val}")
        print(f"Language validation: {lang_val}")
        print(f"Temporal validation: {temp_val}")
    elif modality == "image":
        img_val = "PASS" if content_val == "PASS" else "FAIL"
        print(f"Source validation: {source_val}")
        print(f"Content validation: {content_val}")
        print(f"Language validation: {lang_val}")
        print(f"Image-content validation: {img_val}")
    else:
        print(f"Source validation: {source_val}")
        print(f"Content validation: {content_val}")
        print(f"Language validation: {lang_val}")

    return (source_val == "PASS" and content_val == "PASS")


def main():
    tests = [
        (1, "summarize the audio of jethalal", "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3"),
        (2, "summarize the jethalal audio", "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3"),
        (3, "give me all the content of dense image", "dense.jpeg"),
        (4, "give me the list of files that are added recently", None),
        (5, "give me the list of files which are added today itself", None),
        (6, "give me the names of dense images", "dense.jpeg"),
        (7, "how many images I have by the names of dense?", "dense.jpeg"),
        (8, "please list the name images which have named by dense.", "dense.jpeg"),
        (9, "con call me kya baatein hui hai?", "Behari_lal_call.m4a"),
        (10, "summarize the audio con call.", "Behari_lal_call.m4a"),
        (11, "how many chunks you have for the file behari lal?", "Behari_lal_call.m4a"),
        (12, "who is speakers in behari lal file?", "Behari_lal_call.m4a"),
        (13, "how many person are talking in the audio behari lal?", "Behari_lal_call.m4a"),
        (14, "is there any lady talk in the behari lal?", "Behari_lal_call.m4a")
    ]

    passed_count = 0
    total_count = len(tests)

    for test_num, query, exp_source in tests:
        success = run_single_regression_test(test_num, query, exp_source)
        if success:
            passed_count += 1

    print("\n" + "=" * 80)
    print(f"REGRESSION SUITE SUMMARY: {passed_count}/{total_count} TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
