import uuid
import os
import numpy as np
from app.embeddings.face_embedding import detect_and_embed_faces
from config import DEBUG
from app.storage.lancedb_store import (

    get_face_table,
    store_face_record,
    deduplicate_and_store_faces,
    update_face_record_identity
)

# Configurable face vector cosine distance threshold
# Cosine distance <= 0.35 indicates identical/same person face match
FACE_COSINE_DISTANCE_THRESHOLD = 0.35


def compute_cosine_distance(embedding1, embedding2):
    """
    Computes Cosine distance (1 - cosine_similarity) between two normalized 512-D vectors.
    Range: 0.0 (identical) to 2.0 (opposite).
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    similarity = dot_product / (norm1 * norm2)
    # Cosine distance = 1.0 - cosine_similarity
    return max(0.0, 1.0 - float(similarity))


def analyze_faces_in_image(image_path, threshold=FACE_COSINE_DISTANCE_THRESHOLD):
    """
    Analyzes faces in an image for FACE IDENTIFICATION.

    1. Detects all faces in image using InceptionResnetV1 (512-D normalized embeddings).
    2. Persists EVERY detected face in LanceDB face_embeddings table (with deduplication).
    3. Compares embeddings against registered faces (status == 'known').
    4. Returns persistent face_id along with conversational temp_face_id.

    Returns:
        dict containing faces_detected, embeddings_generated, and face records.
    """
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return {
            "status": "ERROR",
            "message": f"Image file not found: {abs_path}",
            "faces_detected": 0,
            "embeddings_generated": 0,
            "faces": []
        }

    detected_faces = detect_and_embed_faces(abs_path)
    faces_detected_count = len(detected_faces)

    # CASE A: Zero faces detected
    if faces_detected_count == 0:
        return {
            "status": "NO_FACE_DETECTED",
            "message": "I couldn't detect a usable face in this image.",
            "faces_detected": 0,
            "embeddings_generated": 0,
            "faces": []
        }

    # Deduplicate and persist every detected face permanently into LanceDB
    persisted_records = deduplicate_and_store_faces(abs_path, detected_faces)

    # Fetch registered reference records from LanceDB
    ftable = get_face_table()
    df = ftable.to_pandas() if ftable is not None else None

    registered_records = []
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            p_name = str(row.get("person_name", "")).strip()
            p_status = str(row.get("status", "")).strip().lower()
            if p_name and p_name.lower() != "unknown" and p_status == "known":
                registered_records.append(row.to_dict())

    processed_faces = []
    known_count = 0
    unknown_count = 0

    for index, p_rec in enumerate(persisted_records):
        temp_face_id = f"temporary_{index + 1:03d}"
        persistent_face_id = p_rec.get("face_id")
        embedding = p_rec.get("face_embedding")
        bbox = p_rec.get("bbox")
        existing_name = p_rec.get("person_name")
        existing_status = p_rec.get("status")
        existing_src = p_rec.get("identity_source")

        # If already explicitly registered in DB
        if existing_name and existing_name.lower() != "unknown" and existing_status == "known":
            known_count += 1
            processed_faces.append({
                "face_id": persistent_face_id,
                "temp_face_id": temp_face_id,
                "source_id": abs_path,
                "bbox": bbox,
                "embedding": embedding,
                "status": "known",
                "distance": 0.0,
                "person_name": existing_name,
                "identity_source": existing_src if existing_src and existing_src != "none" else "user_registration"
            })
            continue

        best_match_name = None
        min_dist = 2.0
        identity_src = "none"

        # Search registered face memory for closest matching embedding
        for reg in registered_records:
            reg_emb = reg.get("face_embedding")
            if reg_emb is None:
                continue

            dist = compute_cosine_distance(embedding, reg_emb)
            if dist < min_dist:
                min_dist = dist
                best_match_name = reg.get("person_name")

        # Threshold decision for identity provenance
        if best_match_name and min_dist <= threshold:
            status = "known"
            person_name = best_match_name
            identity_src = "face_memory"
            known_count += 1
        else:
            status = "unknown"
            person_name = None
            identity_src = "none"
            unknown_count += 1

        processed_faces.append({
            "face_id": persistent_face_id,
            "temp_face_id": temp_face_id,
            "source_id": abs_path,
            "bbox": bbox,
            "embedding": embedding,
            "status": status,
            "distance": round(float(min_dist), 4),
            "person_name": person_name,
            "identity_source": identity_src
        })

    return {
        "status": "SUCCESS",
        "message": f"Processed {faces_detected_count} face(s).",
        "faces_detected": faces_detected_count,
        "embeddings_generated": len(processed_faces),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "faces": processed_faces
    }


def register_pending_face(pending_face_dict, person_name):
    """
    Registers user-provided person_name for an existing persistent face record in LanceDB.
    Updates the SAME persistent record in LanceDB while preserving original face_id,
    bbox, embedding, and image_path. Sets identity_source = 'user_registration'.
    """
    persistent_face_id = pending_face_dict.get("face_id")
    image_path = pending_face_dict.get("source_id") or pending_face_dict.get("image_path")
    abs_path = os.path.abspath(image_path) if image_path else ""
    bbox = pending_face_dict.get("bbox", [0, 0, 0, 0])
    embedding = pending_face_dict.get("embedding", [])

    if not abs_path:
        return {"status": "ERROR", "message": "Invalid pending face registration data."}

    # Attempt direct update by persistent face_id
    if persistent_face_id:
        updated = update_face_record_identity(persistent_face_id, person_name, identity_source="user_registration")
        if updated:
            return {
                "status": "SUCCESS",
                "person_name": person_name,
                "face_id": persistent_face_id,
                "image_path": abs_path,
                "bbox": bbox,
                "identity_source": "user_registration"
            }

    # Fallback: locate closest matching face record in LanceDB by image_path + bbox/embedding
    ftable = get_face_table()
    df = ftable.to_pandas() if ftable is not None else None

    target_face_id = None
    if df is not None and not df.empty and "image_path" in df.columns:
        image_records = df[df["image_path"] == abs_path]
        if not image_records.empty:
            if embedding and len(embedding) == 512:
                best_dist = 2.0
                for idx, row in image_records.iterrows():
                    rec_emb = row.get("face_embedding")
                    if rec_emb is not None:
                        dist = compute_cosine_distance(embedding, rec_emb)
                        if dist < best_dist:
                            best_dist = dist
                            target_face_id = row.get("face_id")
            else:
                target_face_id = image_records.iloc[0].get("face_id")

    if target_face_id:
        update_face_record_identity(target_face_id, person_name, identity_source="user_registration")
        final_id = target_face_id
    else:
        new_rec = store_face_record(
            face_id=f"face_{uuid.uuid4()}",
            image_path=abs_path,
            bbox=bbox,
            face_embedding=embedding,
            person_name=person_name,
            status="known",
            identity_source="user_registration"
        )
        final_id = new_rec.get("face_id")

    return {
        "status": "SUCCESS",
        "person_name": person_name,
        "face_id": final_id,
        "image_path": abs_path,
        "bbox": bbox,
        "identity_source": "user_registration"
    }



def search_images_by_registered_face(person_name, threshold=FACE_COSINE_DISTANCE_THRESHOLD, max_results=10):
    """
    FACE SEARCH OPERATION:
    Finds all images containing faces matching the registered face embeddings for person_name.
    Uses ONLY face cosine distance comparison against LanceDB face_embeddings table.
    Does NOT use filename matching, CLIP, OCR, or LLM reasoning!
    """
    ftable = get_face_table()
    if ftable is None:
        return []

    df = ftable.to_pandas()
    if df.empty:
        return []

    # Fetch registered reference embeddings for target person
    person_records = df[df["person_name"].str.lower() == person_name.lower()]
    if person_records.empty:
        if DEBUG:
            print(f"No registered face memory record found for person '{person_name}'.")
        return []

    reference_embeddings = person_records["face_embedding"].tolist()

    matching_paths = set()
    matched_results = []

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        target_embedding = row["face_embedding"]

        # Calculate minimum cosine distance to any reference embedding of person_name
        min_dist = min(
            compute_cosine_distance(ref_emb, target_embedding)
            for ref_emb in reference_embeddings
        )

        # Match strictly based on vector distance threshold
        if min_dist <= threshold:
            if img_path not in matching_paths:
                matching_paths.add(img_path)
                rec = row.to_dict()
                rec["distance"] = round(float(min_dist), 4)
                rec["identity_source"] = "face_memory"
                matched_results.append(rec)

        if len(matched_results) >= max_results:
            break

    return matched_results


# Alias for backward compatibility
find_faces_for_person = search_images_by_registered_face



def register_face(person_name, image_path):
    """
    Helper function for backward compatibility registering a reference face from an image path.
    """
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return {"status": "ERROR", "message": f"Image file not found: {abs_path}"}

    faces = detect_and_embed_faces(abs_path)
    if not faces:
        return {"status": "ERROR", "message": f"No face detected in {os.path.basename(abs_path)}"}

    sorted_faces = sorted(faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
    primary_face = sorted_faces[0]

    pending_entry = {
        "source_id": abs_path,
        "bbox": primary_face["bbox"],
        "embedding": primary_face["embedding"]
    }
    return register_pending_face(pending_entry, person_name)

