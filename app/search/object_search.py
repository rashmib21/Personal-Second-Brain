import os
from app.storage.lancedb_store import get_image_table
from app.rag.image_summarizer import summarize_image


def extract_object_from_question(question):
    """
    Extract the object that the user wants to search for.

    Examples:
        "Find images containing a car" -> "car"
        "Find images with a laptop" -> "laptop"
        "Which images contain a dog?" -> "dog"
    """

    question_lower = question.lower().strip()

    patterns = [
        "find images containing ",
        "find images with ",
        "show images containing ",
        "show images with ",
        "images containing ",
        "images with ",
        "which images contain ",
        "which images have ",
        "find a picture of ",
        "find pictures of ",
        "find photos of ",
        "show me images containing ",
        "show me images with ",
    ]

    for pattern in patterns:
        if pattern in question_lower:
            object_name = question_lower.split(pattern, 1)[1]

            # Remove common punctuation
            object_name = object_name.strip(" ?.!,")

            return object_name

    return None


def search_images_by_object(question, max_results=10):
    """
    Search indexed images for a requested object.

    The current implementation uses Qwen2.5-VL
    to inspect each indexed image.
    """

    image_table = get_image_table()

    if image_table is None:
        print("Image table is not available.")
        return []

    image_df = image_table.to_pandas()

    if image_df.empty:
        print("No indexed images found.")
        return []

    object_name = extract_object_from_question(question)

    if not object_name:
        print("Could not extract object from question.")
        return []

    print("\n===== OBJECT SEARCH =====")
    print("Requested object:", object_name)

    results = []

    # Get unique image paths
    image_paths = sorted(
        set(image_df["path"].tolist())
    )

    for image_path in image_paths:

        if not os.path.exists(image_path):
            print("Skipping missing image:", image_path)
            continue

        try:
            prompt = f"""
Look at this image carefully.

Determine whether the image contains this object:

"{object_name}"

Answer ONLY with:
YES
or
NO

Do not explain your answer.
"""

            # We use the existing Qwen2.5-VL function.
            # The function must support a custom question/prompt.
            answer = summarize_image(
                image_path,
                prompt
            )

            answer_clean = answer.strip().upper()

            print(
                os.path.basename(image_path),
                "->",
                answer_clean
            )

            if answer_clean.startswith("YES"):
                # Get the corresponding image record
                image_rows = image_df[
                    image_df["path"] == image_path
                ]

                if not image_rows.empty:
                    result = image_rows.iloc[0].to_dict()
                    results.append(result)

            if len(results) >= max_results:
                break

        except Exception as e:
            print(
                "Object detection failed for",
                os.path.basename(image_path),
                ":",
                str(e)
            )

    print(
        "Images containing",
        object_name,
        ":",
        len(results)
    )

    return results