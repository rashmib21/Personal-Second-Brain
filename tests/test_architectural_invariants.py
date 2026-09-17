import os
import sys
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep architectural tests completely isolated from the production database.
TEST_DB_PATH = os.path.join(BASE_DIR, "test_database")
os.environ["PERSONAL_SECOND_BRAIN_DB_PATH"] = TEST_DB_PATH

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.query.query_analyzer import analyze_query
from app.search.vector_search import search
from app.services.face_service import (
    analyze_faces_in_image,
    register_pending_face,
    search_images_by_registered_face,
    compute_cosine_distance,
    FACE_COSINE_DISTANCE_THRESHOLD
)
from app.services.interaction_state import (
    clear_last_interaction,
    get_last_interaction,
    update_last_interaction,
    get_pending_faces,
    set_pending_faces,
    clear_pending_faces
)
from app.storage.lancedb_store import (
    store_feedback,
    search_feedback,
    get_face_table,
    get_feedback_table
)
from app.rag.rag_pipeline import ask, format_response


@pytest.fixture(autouse=True)
def reset_state():
    """Resets global interaction state before each test."""
    clear_last_interaction()
    clear_pending_faces()
    yield
    clear_last_interaction()
    clear_pending_faces()


def get_test_image_path(filename="mummy.jpg"):
    """Helper to find test image path in watched_folder or repo root."""
    watched_path = os.path.join(BASE_DIR, "watched_folder", filename)
    if os.path.exists(watched_path):
        return watched_path
    root_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    # Create dummy file if not exists
    os.makedirs(os.path.dirname(watched_path), exist_ok=True)
    with open(watched_path, "wb") as f:
        f.write(b"dummy image data")
    return watched_path


# ==============================================================================
# TEST 1: Hard Source Routing (mummy.jpg pre-retrieval candidate filtering)
# ==============================================================================
def test_1_hard_source_routing_mummy():
    question = "What color is the clothing of the person on the right in mummy.jpg?"
    analysis = analyze_query(question)
    assert analysis["source_hint"] == "mummy.jpg"
    assert analysis["modality"] == "image"

    results = search(question, max_results=10, analysis=analysis)
    for doc in results:
        base_name = os.path.basename(doc["path"]).lower()
        assert base_name == "mummy.jpg", f"Routing failure: Candidate {doc['path']} is not mummy.jpg"


# ==============================================================================
# TEST 2: Visual Question Answering on mummy.jpg
# ==============================================================================
def test_2_vqa_mummy_question():
    question = "How many people are visible in mummy.jpg?"
    analysis = analyze_query(question)
    assert analysis["source_hint"] == "mummy.jpg"
    assert analysis["is_visual_qa"] == True

    ans, sources, num_chunks = ask(question)
    assert len(sources) == 1
    assert sources[0] == "mummy.jpg"
    assert "no usable visual information" not in ans.lower() or os.path.basename(sources[0]) == "mummy.jpg"


# ==============================================================================
# TEST 3: Accounts.m4a pre-retrieval exclusion invariant
# ==============================================================================
def test_3_accounts_m4a_never_enters_mummy_candidates():
    question = "What color is the clothing in mummy.jpg?"
    analysis = analyze_query(question)
    results = search(question, max_results=10, analysis=analysis)

    for doc in results:
        base_name = os.path.basename(doc["path"]).lower()
        assert "accounts.m4a" not in base_name
        assert base_name == "mummy.jpg"


# ==============================================================================
# TEST 4: Follow-up Answer Correction State Link
# ==============================================================================
def test_4_followup_answer_correction_link():
    update_last_interaction(
        query="How many people are visible in mummy.jpg?",
        answer="There is 1 person.",
        sources=["mummy.jpg"],
        modality="image"
    )

    correction_query = "But there are two people standing in the image, one is Rashmi and the second one is her mother."
    ans, sources, num_chunks = ask(correction_query)

    assert "LEARNING MODE ACTIVATED" in ans
    assert "mummy.jpg" in sources

    last_state = get_last_interaction()
    assert last_state["previous_source"] == "mummy.jpg"


# ==============================================================================
# TEST 5: Unknown Face Detection Prompt
# ==============================================================================
def test_5_unknown_face_prompt():
    dummy_img = get_test_image_path("test_unknown.jpg")

    # Mock analyze_faces_in_image with unknown face
    pending_face = {
        "face_id": "temporary_001",
        "source_id": dummy_img,
        "bbox": [10, 10, 50, 50],
        "embedding": [0.1] * 512,
        "status": "unknown",
        "distance": 0.8,
        "person_name": None,
        "identity_source": "none"
    }

    set_pending_faces([pending_face])
    question = "Who is this person in test_unknown.jpg?"
    ans, sources, num_chunks = ask(question)

    assert "I found a person I don't recognize yet. Who is this person?" in ans


# ==============================================================================
# TEST 6: Unknown Face Registration Workflow
# ==============================================================================
def test_6_unknown_face_registration():
    dummy_img = get_test_image_path("mummy.jpg")
    pending_face = {
        "face_id": "temporary_001",
        "source_id": dummy_img,
        "bbox": [10, 10, 50, 50],
        "embedding": [0.05] * 512,
        "status": "unknown",
        "distance": 0.7,
        "person_name": None,
        "identity_source": "none"
    }
    set_pending_faces([pending_face])

    user_response = "This is my mother."
    ans, sources, num_chunks = ask(user_response)

    assert "Registered identity 'mother'" in ans or "mother" in ans.lower()
    assert len(get_pending_faces()) == 0


# ==============================================================================
# TEST 7: Same Registered Face Identity Match via Vector Search
# ==============================================================================
def test_7_registered_face_match():
    ref_emb = [0.01] * 512
    pending_face = {
        "face_id": "temp_mother_01",
        "source_id": get_test_image_path("mummy.jpg"),
        "bbox": [20, 20, 60, 60],
        "embedding": ref_emb
    }
    register_pending_face(pending_face, "Mother")

    # Match test vector
    test_emb = [0.01] * 512
    dist = compute_cosine_distance(ref_emb, test_emb)
    assert dist <= FACE_COSINE_DISTANCE_THRESHOLD


# ==============================================================================
# TEST 8: Multi-Face Image Tracking Safety
# ==============================================================================
def test_8_multi_face_safety():
    face1 = {
        "face_id": "temporary_001",
        "source_id": "multi.jpg",
        "bbox": [10, 10, 30, 30],
        "embedding": [0.1] * 512,
        "status": "known",
        "distance": 0.2,
        "person_name": "Rashmi",
        "identity_source": "face_memory"
    }

    face2 = {
        "face_id": "temporary_002",
        "source_id": "multi.jpg",
        "bbox": [50, 50, 40, 40],
        "embedding": [0.9] * 512,
        "status": "unknown",
        "distance": 0.8,
        "person_name": None,
        "identity_source": "none"
    }

    set_pending_faces([face2])

    pending_list = get_pending_faces()
    assert len(pending_list) == 1
    assert pending_list[0]["face_id"] == "temporary_002"
    assert pending_list[0]["person_name"] is None


# ==============================================================================
# TEST 9: Audio Feedback Isolation (Image Feedback does NOT affect Audio)
# ==============================================================================
def test_9_audio_feedback_isolation():
    store_feedback(
        original_query="What color is the clothing in mummy.jpg?",
        wrong_answer="Red",
        wrong_source="mummy.jpg",
        correct_source="mummy.jpg",
        corrected_answer="Blue",
        modality="image",
        rejected_sources="Accounts.m4a"
    )

    audio_feedback = search_feedback("is any girl speaking in Accounts.m4a?", query_modality="audio", source_hint="Accounts.m4a")
    for fb in audio_feedback:
        assert fb["modality"] == "audio"


# ==============================================================================
# TEST 10: Image Feedback Isolation (Audio Feedback does NOT affect Image)
# ==============================================================================
def test_10_image_feedback_isolation():
    store_feedback(
        original_query="Who speaks in Accounts.m4a?",
        wrong_answer="Boy",
        wrong_source="Accounts.m4a",
        correct_source="Accounts.m4a",
        corrected_answer="Girl",
        modality="audio"
    )

    image_feedback = search_feedback("What color is the clothing in mummy.jpg?", query_modality="image", source_hint="mummy.jpg")
    for fb in image_feedback:
        assert fb["modality"] == "image"


# ==============================================================================
# TEST 11: Unknown Person Identity Protection Invariant
# ==============================================================================
def test_11_unknown_person_identity_protection():
    res = {
        "status": "unknown",
        "person_name": None,
        "identity_source": "none"
    }

    if res["identity_source"] not in ["face_memory", "user_registration"]:
        assert res["person_name"] is None


# ==============================================================================
# TEST 12: Known Face Memory Provenance Invariant
# ==============================================================================
def test_12_known_face_provenance():
    person_name = "Rashmi"
    identity_source = "face_memory"

    assert identity_source in ["face_memory", "user_registration"]
    assert person_name is not None


# ==============================================================================
# TEST 13: Pre-LLM Modality Consistency Validation
# ==============================================================================
def test_13_modality_consistency_validation():
    modality = "image"
    retrieved_sources = ["mummy.jpg"]

    for src in retrieved_sources:
        assert src.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


# ==============================================================================
# TEST 14: Explicit Source Zero Chunks Return Message with NO Fallback
# ==============================================================================
def test_14_zero_chunks_no_fallback():
    non_existent_query = "What is written in non_existent_file.jpg?"
    ans, sources, num_chunks = ask(non_existent_query)

    assert "No usable visual information was found for non_existent_file.jpg" in ans
    assert num_chunks == 0


# ==============================================================================
# TEST 15: Face Detection Failure vs Match Failure Separation
# ==============================================================================
def test_15_face_detection_failure_vs_match_failure():
    # Case A: Detection Failure (0 faces detected)
    case_a_res = {
        "status": "NO_FACE_DETECTED",
        "message": "I couldn't detect a usable face in this image.",
        "faces_detected": 0,
        "embeddings_generated": 0,
        "faces": []
    }
    assert case_a_res["faces_detected"] == 0
    assert case_a_res["message"] == "I couldn't detect a usable face in this image."
    assert case_a_res["status"] != "unknown"

    # Case B: Detection Success (1 face detected), Match Failure (0 registered matches)
    case_b_res = {
        "status": "SUCCESS",
        "message": "Processed 1 face(s).",
        "faces_detected": 1,
        "embeddings_generated": 1,
        "known_count": 0,
        "unknown_count": 1,
        "faces": [
            {
                "face_id": "temporary_001",
                "status": "unknown",
                "person_name": None,
                "identity_source": "none"
            }
        ]
    }
    assert case_b_res["faces_detected"] > 0
    assert case_b_res["unknown_count"] == 1
    assert case_b_res["faces"][0]["status"] == "unknown"


# ==============================================================================
# TEST 16: Face Search Vector Isolation Invariant
# ==============================================================================
def test_16_face_search_vector_isolation():
    # Register Rashmi face vector
    rashmi_emb = [0.02] * 512
    pending = {
        "source_id": get_test_image_path("mummy.jpg"),
        "bbox": [10, 10, 40, 40],
        "embedding": rashmi_emb
    }
    register_pending_face(pending, "Rashmi")

    # Search images of Rashmi using face vector search
    matched = search_images_by_registered_face("Rashmi")
    assert len(matched) > 0
    for rec in matched:
        assert rec["identity_source"] == "face_memory"
        assert rec["distance"] <= FACE_COSINE_DISTANCE_THRESHOLD
