import os
import re
from app.storage.lancedb_store import get_face_table, get_image_table
from app.services.face_service import find_faces_for_person


def is_face_search_query(question):
    """
    Detects if a question is a face-memory or person search query.

    Examples:
        - "Which images contain faces?"
        - "Find images containing Rashmi"
        - "Find images of Rashmi"
        - "show me the picture of Rashmi"
        - "I want to see the image of Rashmi"
        - "show me Rashmi's photo"
    """
    question_lower = question.lower().strip()

    phrases = [
        "images contain faces",
        "pictures contain faces",
        "images containing faces",
        "pictures containing faces",
        "images with faces",
        "pictures with faces",
        "find images of",
        "find pictures of",
        "find images containing",
        "find pictures containing",
        "images containing",
        "pictures containing",
        "images of",
        "pictures of",
        "show images of",
        "show pictures of",
        "picture of",
        "image of",
        "photo of",
        "photos of",
        "pictures of",
        "images of",
        "who is in",
        "contain faces",
        "contains faces",
        "have faces",
        "has faces"
    ]

    if any(phrase in question_lower for phrase in phrases):
        return True

    pattern_verb = r"\b(find|show|see|look|which|list)\b.*\b(image|images|picture|pictures|photo|photos|face|faces)\b.*\b(of|containing|with|having)\b"
    if re.search(pattern_verb, question_lower):
        return True

    pattern_possessive = r"\b([a-z0-9_]+)'s\s+(photo|photos|picture|pictures|image|images|face|faces)\b"
    if re.search(pattern_possessive, question_lower):
        return True

    return False


def extract_person_name_from_question(question):
    """
    Extracts the person's name or keyword from a face search question.
    Returns None for generic face search queries (e.g. "images containing faces").
    """
    q = question.lower().strip()

    # Generic face query check
    generic_keywords = [
        "contain faces", "contains faces", "have faces", "has faces",
        "images with faces", "pictures with faces", "containing faces", "with faces",
        "of faces", "all faces", "any faces"
    ]
    if any(k in q for k in generic_keywords):
        return None

    # Check for possessive pattern: "<name>'s photo/picture/image"
    m_possessive = re.search(r"\b([a-z0-9_]+)'s\s+(photo|photos|picture|pictures|image|images|face|faces)\b", q)
    if m_possessive:
        name = m_possessive.group(1).strip()
        if name not in ["faces", "face", "person", "someone", "people", "all"]:
            return name

    # Check for direct verification query patterns: "is Rashmi in mummy.jpg?", "in mummy.jpg there is Rashmi found?", "does mummy.jpg contain Rashmi?"
    m_is_in = re.search(r"\bis\s+([a-z0-9_]+)\s+in\b", q)
    if m_is_in:
        name = m_is_in.group(1).strip()
        if name not in ["there", "this", "that", "it", "he", "she", "anyone", "someone"]:
            return name

    m_found = re.search(r"\bthere\s+is\s+([a-z0-9_]+)\s+(found|present)\b", q)
    if m_found:
        name = m_found.group(1).strip()
        if name not in ["a", "an", "the", "one", "someone", "anyone"]:
            return name

    m_contain = re.search(r"\bcontains?\s+([a-z0-9_]+)\b", q)
    if m_contain:
        name = m_contain.group(1).strip()
        if name not in ["faces", "face", "person", "someone", "people", "all", "a", "an", "the"]:
            return name

    # Check for preposition pattern: "(picture/image/photo) of/containing/with (the/a/an)? <name>"
    m_prep = re.search(r"\b(image|images|picture|pictures|photo|photos|face|faces)\s+(of|containing|with|having)\s+(the\s+|a\s+|an\s+|my\s+)?([a-z0-9_]+)\b", q)
    if m_prep:
        name = m_prep.group(4).strip()
        if name not in ["faces", "face", "person", "someone", "people", "all"]:
            return name

    # String prefix patterns
    patterns = [
        "find all images containing ", "find all pictures containing ", "find images containing ", "find pictures containing ",
        "show me images containing ", "show me pictures containing ", "show images containing ", "show pictures containing ",
        "find images of ", "find pictures of ", "show me images of ", "show me pictures of ", "show images of ", "show pictures of ",
        "show me the picture of ", "show me the image of ", "show me the photo of ",
        "picture of ", "image of ", "photo of ", "photos of ", "pictures of ", "images of "
    ]

    for p in patterns:
        if p in q:
            name = q.split(p, 1)[1].strip(" ?.!,")
            for art in ["the ", "a ", "an ", "my "]:
                if name.startswith(art):
                    name = name[len(art):]
            if name and name.lower() not in ["faces", "face", "person", "someone", "people"]:
                return name

    return None


def search_images_by_face(question, max_results=10):
    """
    Searches LanceDB for images matching a specific person's registered face embeddings using vector distance.
    Enforces strict vector distance thresholding without filename matching or CLIP fallback.
    """
    from app.services.face_service import search_images_by_registered_face

    ftable = get_face_table()
    if ftable is None:
        print("Face table is not available.")
        return []

    person_name = extract_person_name_from_question(question)

    print("\n===== FACE MEMORY SEARCH =====")
    if person_name:
        print(f"Target Person Name: '{person_name}'")
        matched_records = search_images_by_registered_face(person_name, max_results=max_results)

        results = []
        if matched_records:
            for rec in matched_records:
                p = rec.get("image_path") or rec.get("path")
                item = dict(rec)
                item["type"] = "image"
                item["image_path"] = p
                item["path"] = p
                item["source"] = os.path.basename(p) if p else "unknown"
                results.append(item)
            return results

        print(f"No registered face memory record found matching '{person_name}'.")
        return []

    else:
        print("Target Query: Generic Face Search (all images containing faces)")
        df = ftable.to_pandas()
        if df.empty:
            print("No face embeddings indexed yet.")
            return []

        unique_paths = df["image_path"].dropna().unique()
        results = []
        for p in unique_paths[:max_results]:
            count = len(df[df["image_path"] == p])
            results.append({
                "type": "image",
                "image_path": p,
                "path": p,
                "source": os.path.basename(p),
                "face_count": count
            })
        print(f"Found {len(results)} image(s) containing detected faces.")
        return results

