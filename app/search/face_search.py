import os
import re
from app.storage.lancedb_store import get_face_table
from app.services.face_service import find_faces_for_person


def is_face_search_query(question):
    """
    Detects if a question is a face-memory or person search query.

    Examples:
        - "Which images contain faces?"
        - "Find images containing Rashmi"
        - "Find images of Rashmi"
        - "Find all images containing this person"
    """
    question_lower = question.lower().strip()

    phrases = [
        "images contain faces",
        "pictures contain faces",
        "images containing faces",
        "pictures containing faces",
        "images with faces",
        "pictures with faces",
        "find images of ",
        "find pictures of ",
        "find images containing ",
        "find pictures containing ",
        "images containing ",
        "pictures containing ",
        "images of ",
        "pictures of ",
        "show images of ",
        "show pictures of ",
        "who is in ",
        "contain faces",
        "contains faces",
        "have faces",
        "has faces"
    ]

    if any(phrase in question_lower for phrase in phrases):
        return True

    # Regex pattern matching "find/which/show images of <name>" or "images containing <name>"
    pattern = r"\b(find|show|which|list)\b.*\b(image|images|picture|pictures|photo|photos)\b.*\b(of|containing|with|having)\b.*\b(person|[a-z0-9_]+)\b"
    if re.search(pattern, question_lower):
        return True

    return False


def extract_person_name_from_question(question):
    """
    Extracts the person's name or keyword from a face search question.
    """
    question_lower = question.lower().strip()

    # Generic face queries
    if any(k in question_lower for k in ["contain faces", "contains faces", "have faces", "has faces", "images with faces", "pictures with faces"]):
        return None

    patterns = [
        "find all images containing ",
        "find all pictures containing ",
        "find images containing ",
        "find pictures containing ",
        "show me images containing ",
        "show me pictures containing ",
        "find images of ",
        "find pictures of ",
        "show me images of ",
        "show me pictures of ",
        "show images of ",
        "show pictures of ",
        "images containing ",
        "pictures containing ",
        "images of ",
        "pictures of ",
        "photos of ",
        "photo of "
    ]

    for pattern in patterns:
        if pattern in question_lower:
            name = question_lower.split(pattern, 1)[1]
            name = name.strip(" ?.!,")
            if name.lower() in ["faces", "face"]:
                return None
            return name

    return None


def search_images_by_face(question, max_results=10):
    """
    Searches LanceDB for images containing faces or matching a specific person's face embeddings.

    Returns:
        list of dicts containing matching image paths and metadata
    """
    ftable = get_face_table()
    if ftable is None:
        print("Face table is not available.")
        return []

    df = ftable.to_pandas()
    if df.empty:
        print("No face embeddings indexed yet.")
        return []

    person_name = extract_person_name_from_question(question)

    print("\n===== FACE MEMORY SEARCH =====")
    if person_name:
        print(f"Target Person Name: '{person_name}'")
        matched_records = find_faces_for_person(person_name, threshold=0.35, max_results=max_results)
        return matched_records
    else:
        print("Target Query: Generic Face Search (all images containing faces)")
        # Get all unique image paths that have detected face embeddings
        unique_paths = df["image_path"].unique()
        results = []
        for p in unique_paths[:max_results]:
            count = len(df[df["image_path"] == p])
            results.append({
                "image_path": p,
                "face_count": count
            })
        print(f"Found {len(results)} image(s) containing detected faces.")
        return results
