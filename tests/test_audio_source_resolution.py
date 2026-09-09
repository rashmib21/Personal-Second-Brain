import pytest
import os
from app.query.query_analyzer import analyze_query, find_best_matching_source
from app.rag.rag_pipeline import ask


def test_jethalal_audio_source_resolution():
    """
    Test that queries referencing 'jethalal' resolve to the exact indexed Jethalal audio file,
    and do NOT resolve to generic files like 'audio.mpeg'.
    """
    indexed_files = [
        "audio.mpeg",
        "Accounts.m4a",
        "MLKDream.mp3",
        "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3"
    ]

    query_list = [
        "Summarize the audio of jethalal.",
        "Give me a summary of jethalal's audio",
        "What is discussed in jethalal audio?",
        "Summarize jethalal recording",
        "What was said in the jethalal audio?"
    ]

    expected_file = "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3"

    for question in query_list:
        analysis_result = analyze_query(question, indexed_files=indexed_files)
        detected_source = analysis_result.get("source_hint")

        assert detected_source == expected_file, f"Query '{question}' resolved to '{detected_source}' instead of '{expected_file}'"
        assert analysis_result.get("modality") == "audio"


def test_explicit_audio_filenames_resolution():
    """
    Test that explicit filenames like Accounts.m4a and audio.mpeg resolve to themselves.
    """
    analysis_accounts = analyze_query("Summarize Accounts.m4a")
    assert analysis_accounts.get("source_hint") == "Accounts.m4a"
    assert analysis_accounts.get("modality") == "audio"

    analysis_audio_mpeg = analyze_query("Summarize audio.mpeg")
    assert analysis_audio_mpeg.get("source_hint") == "audio.mpeg"
    assert analysis_audio_mpeg.get("modality") == "audio"


def test_unresolved_audio_source():
    """
    Test that an audio summarization query for an unknown entity returns UNRESOLVED_AUDIO_SOURCE
    and does not fall back to an arbitrary file.
    """
    analysis_unknown = analyze_query("Summarize the audio of unknown_person_nonexistent_xyz")
    assert analysis_unknown.get("source_hint") == "UNRESOLVED_AUDIO_SOURCE"

    answer, sources, num_chunks = ask("Summarize the audio of unknown_person_nonexistent_xyz")
    assert "couldn't identify" in answer.lower()
    assert len(sources) == 0
    assert num_chunks == 0


def test_jethalal_audio_summarization_end_to_end():
    """
    End-to-end integration test: Asking 'Summarize the audio of jethalal.'
    verifies that ONLY the Jethalal audio file is retrieved, hard invariant passes,
    and no unrelated sources (e.g. Accounts.m4a) are retrieved.
    """
    question = "Summarize the audio of jethalal."
    answer, sources, num_chunks = ask(question)

    expected_filename = "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3"

    # Step 1: Verify retrieved sources list contains ONLY the Jethalal audio file
    assert len(sources) == 1, f"Expected exactly 1 source, but got {len(sources)}: {sources}"
    assert sources[0] == expected_filename, f"Expected source '{expected_filename}', but got '{sources[0]}'"

    # Step 2: Verify candidate chunk count is greater than 0
    assert num_chunks > 0, f"Expected num_chunks > 0, but got {num_chunks}"

    # Step 3: Verify answer string is non-empty
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
