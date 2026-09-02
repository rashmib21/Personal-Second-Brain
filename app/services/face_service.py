import uuid
import os
import numpy as np
from app.embeddings.face_embedding import detect_and_embed_faces
from app.storage.lancedb_store import get_face_table, store_face_record


def register_face(person_name, image_path):
    """
    Registers a reference face for a person from an image.
    This creates an explicit 'memory' entry in the face_embeddings LanceDB table.

    Returns:
        dict: Summary of registration result
    """
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return {"status": "ERROR", "message": f"Image file not found: {abs_path}"}

    faces = detect_and_embed_faces(abs_path)
    if not faces:
        return {"status": "ERROR", "message": f"No face detected in {os.path.basename(abs_path)}"}

    # Sort detected faces by bounding box area (w * h) in descending order to select largest face
    sorted_faces = sorted(faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
    primary_face = sorted_faces[0]

    ftable = get_face_table()
    df = ftable.to_pandas()

    # Prevent duplicate registrations of the same person on the same image
    if not df.empty:
        matching = df[(df["person_name"].str.lower() == person_name.lower()) & (df["image_path"] == abs_path)]
        if not matching.empty:
            for fid in matching["face_id"].tolist():
                ftable.delete(f"face_id = '{fid}'")

    face_id = str(uuid.uuid4())

    store_face_record(
        face_id=face_id,
        image_path=abs_path,
        bbox=primary_face["bbox"],
        face_embedding=primary_face["embedding"],
        person_name=person_name
    )

    return {
        "status": "SUCCESS",
        "person_name": person_name,
        "face_id": face_id,
        "image_path": abs_path,
        "bbox": primary_face["bbox"],
        "faces_detected": len(faces)
    }


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
    # Cosine distance
    return max(0.0, 1.0 - float(similarity))


def find_faces_for_person(person_name, threshold=0.35, max_results=10):
    """
    Finds all images containing faces that match the registered reference face(s) for person_name.

    Returns:
        list of dicts containing matching face records and source image paths.
    """
    ftable = get_face_table()
    df = ftable.to_pandas()

    if df.empty:
        return []

    # Get registered reference embeddings for target person
    person_records = df[df["person_name"].str.lower() == person_name.lower()]
    if person_records.empty:
        print(f"No registered reference face found for '{person_name}'.")
        return []

    reference_embeddings = person_records["face_embedding"].tolist()

    matching_image_paths = set()
    matched_results = []

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        target_embedding = row["face_embedding"]

        # Calculate minimum distance to any reference embedding of person_name
        min_dist = min(
            compute_cosine_distance(ref_emb, target_embedding)
            for ref_emb in reference_embeddings
        )

        # Match strictly based on vector distance threshold
        if min_dist < threshold:
            if img_path not in matching_image_paths:
                matching_image_paths.add(img_path)
                rec = row.to_dict()
                rec["distance"] = min_dist
                matched_results.append(rec)

        if len(matched_results) >= max_results:
            break

    return matched_results
