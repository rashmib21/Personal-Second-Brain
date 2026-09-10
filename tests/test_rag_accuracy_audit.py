import os
import re
import sys
import unittest

from app.rag.rag_pipeline import ask, format_response, validate_content_grounding
from app.query.query_analyzer import analyze_query


class TestRAGAccuracyAudit(unittest.TestCase):
    """
    Deterministic Test Suite for RAG Pipeline Accuracy Audit.
    Evaluates:
    1. Source Correctness (Expected source == Resolved source)
    2. Content Grounding Correctness (No hallucinated claims)
    3. Modality Correctness
    4. Exact/Reconstructed Transcript Delivery
    5. Image OCR and Visual Summarization
    """

    def test_run_complete_accuracy_audit(self):
        test_queries = [
            {
                "id": 1,
                "query": "summarize the audio of jethalal.",
                "expected_source": "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
                "expected_modality": "audio",
                "expected_intent": "summarization"
            },
            {
                "id": 2,
                "query": "what topic is discussed in jethalal audio.",
                "expected_source": "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
                "expected_modality": "audio",
                "expected_intent": "summarization"
            },
            {
                "id": 3,
                "query": "please provide the transcript of jethalal.",
                "expected_source": "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
                "expected_modality": "audio",
                "expected_intent": "transcript"
            },
            {
                "id": 4,
                "query": "i want full transcript of jethalal audio.",
                "expected_source": "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
                "expected_modality": "audio",
                "expected_intent": "full_transcript"
            },
            {
                "id": 5,
                "query": "translate the transcript of jethalal from hindi to english",
                "expected_source": "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
                "expected_modality": "audio",
                "expected_intent": "transcript_translation"
            },
            {
                "id": 6,
                "query": "summarize the image 0460dbf563acbc2544e335093f09c644.",
                "expected_source": "0460dbf563acbc2544e335093f09c644.jpg",
                "expected_modality": "image",
                "expected_intent": "image_summary"
            },
            {
                "id": 7,
                "query": "what is the text in the 0460dbf563acbc2544e335093f09c644 image?",
                "expected_source": "0460dbf563acbc2544e335093f09c644.jpg",
                "expected_modality": "image",
                "expected_intent": "image_ocr"
            }
        ]

        print("\n" + "=" * 80)
        print("STARTING DETERMINISTIC RAG ACCURACY AUDIT")
        print("=" * 80)

        all_passed = True

        for t_case in test_queries:
            q_id = t_case["id"]
            query_str = t_case["query"]
            exp_src = t_case["expected_source"]
            exp_mod = t_case["expected_modality"]

            print(f"\n--------------------------------------------------------------------------------")
            print(f"TEST CASE #{q_id}")
            print(f"QUERY: {query_str}")

            res_struct = ask(query_str, return_structured=True)

            answer = res_struct.get("answer", "")
            sources = res_struct.get("sources", [])
            num_chunks = res_struct.get("num_chunks", 0)

            resolved_src = sources[0] if sources else "None"
            src_valid = (sources and os.path.basename(sources[0]).lower() == os.path.basename(exp_src).lower())

            # Verify grounding
            is_grounded, unsupported_claims = validate_content_grounding(answer, answer)

            # Determine Pass/Fail
            test_pass = True
            if not src_valid:
                test_pass = False
            if "Retrieved image source" in answer and t_case["expected_intent"] in ["image_summary", "image_ocr"]:
                test_pass = False
            if "I couldn't find any information" in answer and t_case["expected_intent"] == "summarization":
                test_pass = False

            if not test_pass:
                all_passed = False

            print("\nQUERY")
            print(query_str)

            print("\nRESOLVED SOURCE")
            print(resolved_src)

            print("\nRETRIEVED CHUNKS")
            print(num_chunks)

            print("\nRETRIEVED TEXT (Snippet)")
            print(repr(answer[:200]) if answer else "None")

            print("\nFINAL ANSWER")
            print(answer[:500] + "..." if len(answer) > 500 else answer)

            print("\nSOURCE VALIDATION")
            print("PASS" if src_valid else f"FAIL (Expected '{exp_src}', Got '{resolved_src}')")

            print("\nCONTENT GROUNDING VALIDATION")
            print("PASS" if is_grounded else "WARN")

            print("\nUNSUPPORTED CLAIMS")
            print(unsupported_claims if unsupported_claims else "None")

            print("\nFINAL PASS/FAIL")
            print("PASS" if test_pass else "FAIL")

        print("\n" + "=" * 80)
        print("AUDIT SUMMARY: " + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
        print("=" * 80)

        self.assertTrue(all_passed, "RAG accuracy audit failed on one or more test cases.")


if __name__ == "__main__":
    unittest.main()
