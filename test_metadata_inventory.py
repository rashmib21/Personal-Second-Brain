import os
import sys
from datetime import datetime, timedelta

from app.utils.date_parser import parse_date_expression
from app.query.query_analyzer import analyze_query
from app.rag.rag_pipeline import ask, handle_temporal_query


def test_date_parser():
    print("\n--- Testing Date Parser ---")
    reference_now = datetime(2026, 9, 11, 17, 30, 0)
    
    # 1. Today
    start_dt, end_dt, label = parse_date_expression("files added today", now=reference_now)
    assert start_dt == datetime(2026, 9, 11, 0, 0, 0)
    assert end_dt == datetime(2026, 9, 11, 23, 59, 59, 999999)
    assert "today" in label
    print("PASS: Today parsing")

    # 2. Yesterday
    start_dt, end_dt, label = parse_date_expression("files uploaded yesterday", now=reference_now)
    assert start_dt == datetime(2026, 9, 10, 0, 0, 0)
    assert end_dt == datetime(2026, 9, 10, 23, 59, 59, 999999)
    assert "yesterday" in label
    print("PASS: Yesterday parsing")

    # 3. N days ago
    start_dt, end_dt, label = parse_date_expression("files added 2 days ago", now=reference_now)
    assert start_dt == datetime(2026, 9, 9, 0, 0, 0)
    assert end_dt == datetime(2026, 9, 9, 23, 59, 59, 999999)
    assert "2 days ago" in label
    print("PASS: N days ago parsing")

    # 4. Specific Month Day (September 9)
    start_dt, end_dt, label = parse_date_expression("files added on September 9", now=reference_now)
    assert start_dt == datetime(2026, 9, 9, 0, 0, 0)
    assert end_dt == datetime(2026, 9, 9, 23, 59, 59, 999999)
    assert "September 09, 2026" in label
    print("PASS: Specific Month Day parsing")

    # 5. All time / No date expression
    start_dt, end_dt, label = parse_date_expression("list all files", now=reference_now)
    assert start_dt is None
    assert end_dt is None
    assert label == "all time"
    print("PASS: All time / No date parsing")


def test_query_analyzer_metadata_classification():
    print("\n--- Testing Query Analyzer Metadata Classification ---")
    
    # 1. List all files
    analysis = analyze_query("list all files")
    assert analysis["temporal_intent"] == "TEMPORAL_FILE_QUERY"
    assert analysis["modality"] == "all"
    assert analysis["extracted_limit"] is None
    print("PASS: list all files classification")

    # 2. Files uploaded yesterday
    analysis = analyze_query("which files were uploaded yesterday")
    assert analysis["temporal_intent"] == "TEMPORAL_FILE_QUERY"
    assert analysis["start_datetime"] is not None
    assert analysis["start_datetime"].day == 10  # yesterday relative to 2026-09-11
    print("PASS: yesterday files classification")

    # 3. Audio files uploaded today
    analysis = analyze_query("which audio files were uploaded today")
    assert analysis["temporal_intent"] == "TEMPORAL_FILE_QUERY"
    assert analysis["modality"] == "audio"
    assert analysis["start_datetime"].day == 11
    print("PASS: today audio files classification")

    # 4. Latest 5 files
    analysis = analyze_query("latest 5 files")
    assert analysis["temporal_intent"] == "TEMPORAL_FILE_QUERY"
    assert analysis["extracted_limit"] == 5
    print("PASS: latest 5 files classification")

    # 5. How many audio files were uploaded yesterday
    analysis = analyze_query("how many audio files were uploaded yesterday")
    assert analysis["temporal_intent"] == "FILE_COUNT"
    assert analysis["modality"] == "audio"
    print("PASS: how many audio files classification")


def test_metadata_rag_pipeline_execution():
    print("\n--- Testing Metadata RAG Pipeline End-to-End Execution ---")
    
    test_queries = [
        "list all files",
        "show all files",
        "which files were uploaded yesterday",
        "which audio files were uploaded yesterday",
        "which files were uploaded today",
        "which images were uploaded today",
        "files added on September 9",
        "files added 2 days ago",
        "latest 5 files",
        "5 recent images",
        "how many audio files were uploaded yesterday"
    ]

    for question in test_queries:
        answer_string, source_list, result_count = ask(question)
        print(f"\nQUERY: {question}")
        print(f"ANSWER:\n{answer_string}")
        print(f"SOURCES: {source_list}")
        print(f"COUNT: {result_count}")

        # Verification constraints
        assert isinstance(answer_string, str)
        assert len(answer_string) > 0
        assert isinstance(source_list, list)
        assert isinstance(result_count, int)
        
        # Metadata queries must never fail with LLM hallucination or crash
        assert "HTTP Request:" not in answer_string
        assert "Error:" not in answer_string


if __name__ == "__main__":
    print("==========================================")
    print("RUNNING METADATA INVENTORY TEST SUITE")
    print("==========================================")
    test_date_parser()
    test_query_analyzer_metadata_classification()
    test_metadata_rag_pipeline_execution()
    print("\nALL METADATA INVENTORY TESTS PASSED SUCCESSFULLY!")
