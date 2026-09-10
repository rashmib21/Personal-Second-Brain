import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.rag.rag_pipeline import ask, format_response
from app.services.interaction_state import clear_last_interaction, clear_pending_faces, get_pending_faces


@pytest.fixture(autouse=True)
def reset_state():
    """Resets interaction state before each test."""
    clear_last_interaction()
    clear_pending_faces()
    yield
    clear_last_interaction()
    clear_pending_faces()


def test_1_show_image_of_my_mummy(capsys):
    """
    Test 1: 'can you please show me the image of my mummy?'
    Should route to source_image_retrieval for mummy.jpg.
    """
    question = "can you please show me the image of my mummy?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: source_image_retrieval" in captured
    assert "mummy.jpg" in sources
    assert "No registered face memory record found matching" not in answer

    formatted = format_response(question, answer, sources, num_chunks)
    assert "SOURCE\nmummy.jpg" in formatted
    assert "PATH\n" in formatted
    assert "mummy.jpg" in formatted


def test_2_show_mummy_jpg(capsys):
    """
    Test 2: 'show me mummy.jpg'
    Should route to source_image_retrieval for mummy.jpg.
    """
    question = "show me mummy.jpg"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: source_image_retrieval" in captured
    assert "mummy.jpg" in sources

    formatted = format_response(question, answer, sources, num_chunks)
    assert "SOURCE\nmummy.jpg" in formatted


def test_3_show_image_of_rashmi(capsys):
    """
    Test 3: 'can you please show me the image of Rashmi?'
    Should route to face_memory_search.
    """
    question = "can you please show me the image of Rashmi?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: face_memory_search" in captured


def test_4_show_photos_of_rashmi(capsys):
    """
    Test 4: 'show me photos of Rashmi'
    Should route to face_memory_search.
    """
    question = "show me photos of Rashmi"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: face_memory_search" in captured


def test_5_is_rashmi_in_mummy_jpg(capsys):
    """
    Test 5: 'is Rashmi in mummy.jpg?'
    Should route to face_verification against mummy.jpg.
    """
    question = "is Rashmi in mummy.jpg?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: face_verification" in captured
    assert "mummy.jpg" in sources


def test_6_does_mummy_jpg_contain_rashmi(capsys):
    """
    Test 6: 'does mummy.jpg contain Rashmi?'
    Should route to face_verification against mummy.jpg.
    """
    question = "does mummy.jpg contain Rashmi?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: face_verification" in captured
    assert "mummy.jpg" in sources


def test_7_no_automatic_registration_on_statement(capsys):
    """
    Test 7: 'so another person in mummy.jpg is my mother'
    Should NOT automatically register or identify unknown face without explicit command.
    """
    question = "so another person in mummy.jpg is my mother"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "FACE_REGISTRATION" not in captured
    assert "Registered identity" not in answer


def test_8_show_image_of_my_mom(capsys):
    """
    Test 8: 'can you please show me the image of my mom?'
    Should route to source_image_retrieval for mummy.jpg.
    """
    question = "can you please show me the image of my mom?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: source_image_retrieval" in captured
    assert "mummy.jpg" in sources


def test_9_show_image_of_my_mother(capsys):
    """
    Test 9: 'can you please show me the image of my mother?'
    Should route to source_image_retrieval for mummy.jpg.
    """
    question = "can you please show me the image of my mother?"
    answer, sources, num_chunks = ask(question)

    captured = capsys.readouterr().out
    assert "Route: source_image_retrieval" in captured
    assert "mummy.jpg" in sources
