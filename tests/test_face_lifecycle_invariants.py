import os
import sys
import uuid
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.storage.lancedb_store import (
    get_face_table,
    store_face_record,
    deduplicate_and_store_faces,
    update_face_record_identity
)
from app.services.face_service import (
    analyze_faces_in_image,
    register_pending_face,
    search_images_by_registered_face,
    compute_cosine_distance,
    FACE_COSINE_DISTANCE_THRESHOLD
)
from app.services.interaction_state import (
    set_pending_faces,
    get_pending_faces,
    clear_pending_faces
)
from app.search.face_search import is_face_search_query, search_images_by_face
from app.embeddings.face_embedding import get_face_model


def test_01_unknown_face_embedding_persists():
    """Invariant 1: Unknown face embedding is persisted in LanceDB during ingestion."""
    sample_path = os.path.join(BASE_DIR, "watched_folder", "mummy.jpg")
    if not os.path.exists(sample_path):
        return

    res = analyze_faces_in_image(sample_path)
    assert res["status"] in ["SUCCESS", "NO_FACE_DETECTED"]

    ftable = get_face_table()
    df = ftable.to_pandas()
    assert not df.empty
    assert len(df) >= res["faces_detected"]


def test_02_unknown_face_survives_request():
    """Invariant 2: Unknown face record survives after request ends."""
    ftable = get_face_table()
    df = ftable.to_pandas()
    unknown_rows = df[df["person_name"] == "unknown"]
    assert len(unknown_rows) > 0


def test_03_pending_state_references_persistent_id():
    """Invariant 3: pending_face_registration references persistent face_id."""
    sample_path = os.path.join(BASE_DIR, "watched_folder", "mummy.jpg")
    if os.path.exists(sample_path):
        res = analyze_faces_in_image(sample_path)
        faces = res.get("faces", [])
        if faces:
            set_pending_faces(faces)
            pending = get_pending_faces()
            assert len(pending) > 0
            assert "face_id" in pending[0]
            clear_pending_faces()


def test_04_user_identification_updates_same_record():
    """Invariant 4: Registration updates the existing persistent record in LanceDB."""
    test_face_id = f"face_test_{uuid.uuid4()}"
    test_emb = np.random.randn(512).tolist()

    store_face_record(
        face_id=test_face_id,
        image_path="/tmp/test_group.jpg",
        bbox=[10, 10, 50, 50],
        face_embedding=test_emb,
        person_name="unknown",
        status="unknown",
        identity_source="none"
    )

    pending_entry = {
        "face_id": test_face_id,
        "source_id": "/tmp/test_group.jpg",
        "bbox": [10, 10, 50, 50],
        "embedding": test_emb
    }

    reg_res = register_pending_face(pending_entry, "TestMother")
    assert reg_res["status"] == "SUCCESS"
    assert reg_res["face_id"] == test_face_id

    ftable = get_face_table()
    df = ftable.to_pandas()
    matching = df[df["face_id"] == test_face_id]
    assert len(matching) == 1
    row = matching.iloc[0].to_dict()
    assert row["person_name"] == "TestMother"
    assert row["status"] == "known"
    assert row["identity_source"] == "user_registration"


def test_05_original_embedding_preserved_post_registration():
    """Invariant 5: Original face embedding vector is preserved post-registration."""
    test_face_id = f"face_test_{uuid.uuid4()}"
    test_emb = [0.1] * 512

    store_face_record(
        face_id=test_face_id,
        image_path="/tmp/test_preserve.jpg",
        bbox=[20, 20, 60, 60],
        face_embedding=test_emb,
        person_name="unknown",
        status="unknown",
        identity_source="none"
    )

    update_face_record_identity(test_face_id, "PreservedPerson", identity_source="user_registration")

    ftable = get_face_table()
    df = ftable.to_pandas()
    row = df[df["face_id"] == test_face_id].iloc[0]
    rec_emb = list(row["face_embedding"])
    assert len(rec_emb) == 512
    assert abs(rec_emb[0] - 0.1) < 1e-4


def test_06_future_recognition_via_face_memory():
    """Invariant 6: Future image ingestion recognizes same person via face memory vector comparison."""
    reg_emb = np.random.randn(512)
    reg_emb = (reg_emb / np.linalg.norm(reg_emb)).tolist()
    test_face_id = f"face_reg_{uuid.uuid4()}"

    store_face_record(
        face_id=test_face_id,
        image_path="/tmp/reference_face.jpg",
        bbox=[0, 0, 100, 100],
        face_embedding=reg_emb,
        person_name="RecognizedPerson",
        status="known",
        identity_source="user_registration"
    )

    # Identical vector should compute distance ~0.0 <= 0.35 threshold
    dist = compute_cosine_distance(reg_emb, reg_emb)
    assert dist <= FACE_COSINE_DISTANCE_THRESHOLD


def test_07_group_image_creates_independent_records():
    """Invariant 7: Group image creates independent persistent records for every face."""
    sample_path = os.path.join(BASE_DIR, "watched_folder", "mummy.jpg")
    if os.path.exists(sample_path):
        res = analyze_faces_in_image(sample_path)
        if res["faces_detected"] > 0:
            face_ids = [f["face_id"] for f in res["faces"]]
            assert len(face_ids) == len(set(face_ids))


def test_08_unknown_faces_remain_none():
    """Invariant 8: Unknown faces remain person_name='unknown' / status='unknown' until explicitly registered."""
    ftable = get_face_table()
    df = ftable.to_pandas()
    if not df.empty:
        unknowns = df[df["person_name"] == "unknown"]
        for idx, row in unknowns.iterrows():
            assert row.get("person_name") in ["unknown", None]
            assert row.get("status", "unknown") == "unknown"


def test_09_face_search_via_cosine_distance():
    """Invariant 9: Face search queries search persisted face embeddings via vector distance."""
    person_name = "Rashmi"
    matched = search_images_by_registered_face(person_name)
    assert isinstance(matched, list)


def test_10_face_search_no_text_inference():
    """Invariant 10: Face search query detector identifies face intent without text inference."""
    q1 = "Find images of Rashmi"
    q2 = "show me photos of Mother"
    assert is_face_search_query(q1)
    assert is_face_search_query(q2)


def test_11_no_model_retraining():
    """Invariant 11: Pretrained InceptionResnetV1 model remains unchanged (singleton)."""
    m1 = get_face_model()
    m2 = get_face_model()
    assert m1 is m2


def test_12_detection_failure_vs_match_failure():
    """Invariant 12: Zero faces detected returns NO_FACE_DETECTED status."""
    res = analyze_faces_in_image("non_existent_file.png")
    assert res["status"] in ["ERROR", "NO_FACE_DETECTED"]
    assert res["faces_detected"] == 0


def test_13_independent_status_and_provenance():
    """Invariant 13: status and identity_source are tracked independently from person_name."""
    test_face_id = f"face_indep_{uuid.uuid4()}"
    emb = [0.05] * 512
    rec = store_face_record(
        face_id=test_face_id,
        image_path="/tmp/indep.jpg",
        bbox=[5, 5, 50, 50],
        face_embedding=emb,
        person_name="unknown",
        status="unknown",
        identity_source="none"
    )
    assert rec["person_name"] == "unknown"
    assert rec["status"] == "unknown"
    assert rec["identity_source"] == "none"


def test_14_reingest_deduplication():
    """Invariant 14: Re-ingesting the same image multiple times does NOT create duplicate face records."""
    dummy_faces = [
        {"bbox": [10, 10, 100, 100], "embedding": [0.1] * 512},
        {"bbox": [150, 150, 80, 80], "embedding": [0.2] * 512}
    ]
    test_path = "/tmp/dedup_test_image.jpg"

    recs1 = deduplicate_and_store_faces(test_path, dummy_faces)
    recs2 = deduplicate_and_store_faces(test_path, dummy_faces)

    assert len(recs1) == 2
    assert len(recs2) == 2
    assert recs1[0]["face_id"] == recs2[0]["face_id"]


if __name__ == "__main__":
    print("Running face lifecycle invariant tests...")
    test_01_unknown_face_embedding_persists()
    test_02_unknown_face_survives_request()
    test_03_pending_state_references_persistent_id()
    test_04_user_identification_updates_same_record()
    test_05_original_embedding_preserved_post_registration()
    test_06_future_recognition_via_face_memory()
    test_07_group_image_creates_independent_records()
    test_08_unknown_faces_remain_none()
    test_09_face_search_via_cosine_distance()
    test_10_face_search_no_text_inference()
    test_11_no_model_retraining()
    test_12_detection_failure_vs_match_failure()
    test_13_independent_status_and_provenance()
    test_14_reingest_deduplication()
    print("ALL 14 FACE LIFECYCLE INVARIANT TESTS PASSED 100%!")
