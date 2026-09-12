import os
import sys
import json
from datetime import datetime

from app.rag.rag_pipeline import ask, handle_temporal_query
from app.query.query_analyzer import analyze_query

# Test Cases Organized by Category
TEST_CATEGORIES = {
    "AUDIO FILE LISTING / METADATA": [
        "list all audio files",
        "show me all audio recordings",
        "what audio files do I have?",
        "give me the names of my audio files",
        "which recordings are in my folder?",
        "show every sound file in inventory"  # Unseen
    ],
    "AUDIO SEARCH / SOURCE RESOLUTION": [
        "find the audio about the conference call",
        "which audio contains the management discussion?",
        "find the recording mentioning inventory",
        "show me the audio related to jethalal",
        "find the recording with the call discussion",
        "which recording is about dream speech"  # Unseen
    ],
    "AUDIO SUMMARY": [
        "summarize the conference call",
        "what was discussed in the call?",
        "what were the main points discussed in the audio?",
        "tell me what happened in the recording",
        "give me a summary of the entire recording",
        "brief summary of Behari_lal_call.m4a"  # Unseen
    ],
    "COMPLETE TRANSCRIPT": [
        "give me the complete transcript of the audio",
        "show me all the spoken content from the recording",
        "give me the entire transcript without summarizing",
        "give me everything said in this audio",
        "show the full spoken transcript in order",
        "full transcript of Accounts.m4a"  # Unseen
    ],
    "AUDIO TRANSLATION": [
        "convert the whole audio transcript into English",
        "translate the complete recording into English",
        "translate every spoken line into English",
        "give me all the audio content in English",
        "convert everything spoken in this recording to English",
        "translate all text of Behari_lal_call.m4a into English"  # Unseen
    ],
    "SOURCE-SPECIFIC TESTS": [
        "what was said in Accounts.m4a?",
        "give me the complete transcript of Accounts.m4a",
        "translate Accounts.m4a into English",
        "summarize Accounts.m4a",
        "what topics were discussed in Accounts.m4a?",
        "summarize Behari_lal_call.m4a"  # Unseen
    ],
    "HINGLISH / NATURAL LANGUAGE": [
        "con call me kya baat hui?",
        "audio me kya discussion hua?",
        "is recording me kya bola gaya hai?",
        "poore audio me kya baatein hui hain?",
        "is call ka complete transcript do",
        "audio ko English me convert karo",
        "Behari lal call me kya discussion hua"  # Unseen
    ],
    "ASR GROUNDING": [
        "what exact words were spoken in the recording?",
        "give me the exact transcript without correcting the speech",
        "which names are actually mentioned in the audio?",
        "which numbers are spoken in the recording?",
        "what percentages were mentioned?",
        "what exact numbers were stated in Accounts.m4a?"  # Unseen
    ],
    "SPEAKER SAFETY": [
        "how many people are speaking in the recording?",
        "who are the speakers?",
        "what are the names of the speakers?",
        "can you identify the speakers in this audio?",
        "who speaks in Behari_lal_call.m4a?"  # Unseen
    ],
    "AUDIO + METADATA": [
        "which audio files were added today?",
        "which audio files were added yesterday?",
        "show my recent audio files",
        "give me the latest 3 audio files",
        "which audio recordings were added 2 days ago?",
        "top 2 audio recordings"  # Unseen
    ],
    "UNSEEN / EDGE CASES": [
        "which audio files were uploaded on January 1 2020?",
        "summarize non_existent_file_999.mp3",
        "what was said in the audio of astronaut?",
        "convert all text of MLKDream.mp3 in english"
    ]
}


def run_audio_pipeline_tests():
    print("==================================================")
    print("STARTING AUDIO PIPELINE COMPREHENSIVE TEST SUITE")
    print("==================================================")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    test_results = []

    for category, queries in TEST_CATEGORIES.items():
        print(f"\n==================================================")
        print(f"CATEGORY: {category}")
        print(f"==================================================")

        for query in queries:
            total_tests += 1
            print(f"\n--------------------------------------------------")
            print(f"Test #{total_tests}: '{query}'")
            
            # Analyze query intent & source resolution first
            analysis = analyze_query(query)
            intent = analysis.get("intent")
            temporal_intent = analysis.get("temporal_intent")
            modality = analysis.get("modality")
            source_hint = analysis.get("source_hint")
            canonical_source_id = analysis.get("canonical_source_id")
            request_scope = analysis.get("request_scope")

            print(f"  Classification:")
            print(f"    - intent: {intent}")
            print(f"    - temporal_intent: {temporal_intent}")
            print(f"    - modality: {modality}")
            print(f"    - request_scope: {request_scope}")
            print(f"    - source_hint: {source_hint}")
            print(f"    - canonical_source_id: {canonical_source_id}")

            # Execute RAG query
            try:
                answer, sources, num_chunks = ask(query)
                
                print(f"  Execution Output:")
                print(f"    - sources ({len(sources)}): {sources}")
                print(f"    - num_chunks: {num_chunks}")
                answer_preview = answer.replace('\n', ' ')[:150]
                print(f"    - answer preview: {answer_preview}...")

                # Evaluation criteria per category
                status = "PASS"
                failure_reason = ""

                # Category 1 & 10: Metadata / File Listing
                if category in ["AUDIO FILE LISTING / METADATA", "AUDIO + METADATA"]:
                    if temporal_intent == "none" and intent not in ["TEMPORAL_FILE_QUERY", "FILE_LIST", "FILE_COUNT"]:
                        status = "FAIL"
                        failure_reason = f"Metadata query failed intent routing (got intent={intent}, temporal_intent={temporal_intent})"
                    elif "January 1 2020" in query:
                        # Zero result check
                        if result_count := len(sources) > 0:
                            status = "FAIL"
                            failure_reason = f"Zero result date query returned {result_count} sources instead of 0."

                # Category 4 & 5: Complete Transcript & Complete Translation
                elif category in ["COMPLETE TRANSCRIPT", "AUDIO TRANSLATION"] or "complete transcript" in query or "convert all" in query or "translate complete" in query:
                    if request_scope != "complete_file":
                        status = "FAIL"
                        failure_reason = f"Complete file request classified scope as '{request_scope}' instead of 'complete_file'."
                    elif canonical_source_id and num_chunks < 2 and "Behari_lal_call" in str(canonical_source_id):
                        status = "FAIL"
                        failure_reason = f"Complete file request fetched only {num_chunks} chunks instead of ALL source chunks."

                # Category 6: Source-Specific Isolation
                elif "Accounts.m4a" in query:
                    if sources and any("Accounts.m4a" not in s for s in sources):
                        status = "FAIL"
                        failure_reason = f"Source-specific query leaked non-Accounts files in sources: {sources}"

                # Category 11: Unresolved Source
                elif "non_existent_file_999.mp3" in query:
                    if sources:
                        status = "FAIL"
                        failure_reason = f"Unresolved source query returned sources: {sources}"
                    elif "not found" not in answer.lower() and "unresolved" not in answer.lower():
                        status = "FAIL"
                        failure_reason = "Unresolved source query did not explicitly state file was not found."

                if status == "PASS":
                    passed_tests += 1
                    print(f"  RESULT: PASS")
                else:
                    failed_tests += 1
                    print(f"  RESULT: FAIL -> {failure_reason}")

                test_results.append({
                    "id": total_tests,
                    "category": category,
                    "query": query,
                    "status": status,
                    "failure_reason": failure_reason,
                    "intent": intent,
                    "temporal_intent": temporal_intent,
                    "modality": modality,
                    "sources": sources,
                    "num_chunks": num_chunks,
                    "answer_snippet": answer[:200]
                })

            except Exception as exc:
                failed_tests += 1
                print(f"  RESULT: EXCEPTION -> {str(exc)}")
                test_results.append({
                    "id": total_tests,
                    "category": category,
                    "query": query,
                    "status": "FAIL",
                    "failure_reason": f"Exception raised: {str(exc)}",
                    "intent": intent,
                    "sources": [],
                    "num_chunks": 0,
                    "answer_snippet": ""
                })

    print("\n==================================================")
    print("AUDIO PIPELINE TEST SUITE SUMMARY")
    print("==================================================")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass Rate: {(passed_tests / total_tests) * 100:.1f}%")

    with open("audio_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\nDetailed results saved to audio_test_results.json")
    return test_results


if __name__ == "__main__":
    run_audio_pipeline_tests()
