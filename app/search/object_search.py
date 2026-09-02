import os
from app.storage.lancedb_store import get_image_table
from app.rag.image_summarizer import summarize_image


def extract_visual_condition_from_question(question):
    """
    Extract the visual condition or object description from a question.

    Examples:
        "Find images containing a person" -> "a person"
        "Which image contains dogs and cats?" -> "dogs and cats"
        "List the name of image in which dogs and cats are playing together" -> "dogs and cats are playing together"
        "Which image shows a person riding a bicycle?" -> "a person riding a bicycle"
    """

    question_lower = question.lower().strip()

    patterns = [
        "list the names of images in which ",
        "list the name of image in which ",
        "list the names of images where ",
        "list the name of image where ",
        "list the names of images with ",
        "list the name of image with ",
        "list the images in which ",
        "list the image in which ",
        "list the images showing ",
        "list the image showing ",
        "list images in which ",
        "list image in which ",
        "list images showing ",
        "list image showing ",
        "list images where ",
        "list image where ",
        "list images with ",
        "list image with ",
        "show me images containing ",
        "show me images with ",
        "show me image containing ",
        "show me image with ",
        "find images containing ",
        "find image containing ",
        "find images showing ",
        "find image showing ",
        "find images with ",
        "find image with ",
        "find images where ",
        "find image where ",
        "show images containing ",
        "show image containing ",
        "show images showing ",
        "show image showing ",
        "show images with ",
        "show image with ",
        "show images where ",
        "show image where ",
        "which images contain ",
        "which image contains ",
        "which images have ",
        "which image has ",
        "which images show ",
        "which image shows ",
        "find a picture of ",
        "find pictures of ",
        "find photos of ",
        "find a photo of ",
        "images containing ",
        "image containing ",
        "images showing ",
        "image showing ",
        "images in which ",
        "image in which ",
        "images where ",
        "image where ",
        "images with ",
        "image with ",
        "picture of ",
        "pictures of ",
        "photo of ",
        "photos of "
    ]

    for pattern in patterns:
        if pattern in question_lower:
            condition = question_lower.split(pattern, 1)[1]
            return condition.strip(" ?.!,")

    return question_lower.strip(" ?.!,")


def extract_object_from_question(question):
    """
    Backward-compatible wrapper for extract_visual_condition_from_question.
    """
    return extract_visual_condition_from_question(question)


def search_images_by_object(question, max_results=10):
    """
    Search indexed images for a requested object or visual condition.

    The current implementation uses Qwen2.5-VL
    to inspect each candidate indexed image.
    """

    image_table = get_image_table()

    if image_table is None:
        print("Image table is not available.")
        return []

    image_df = image_table.to_pandas()

    if image_df.empty:
        print("No indexed images found.")
        return []

    visual_condition = extract_visual_condition_from_question(question)

    if not visual_condition:
        print("Could not extract visual condition from question.")
        return []

    print("\n===== OBJECT / VISUAL CONDITION SEARCH =====")
    print("Requested condition:", visual_condition)

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

Determine whether the image satisfies the following visual condition:

"{visual_condition}"

Answer ONLY with:
YES
or
NO

Do not explain your answer.
"""

            # We use the existing Qwen2.5-VL function.
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

            words = [w.strip(".,!?:;") for w in answer_clean.split()]
            if answer_clean.startswith("YES") or "YES" in words:
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
                "Visual condition detection failed for",
                os.path.basename(image_path),
                ":",
                str(e)
            )

    print(
        "Images matching condition ('" + visual_condition + "'):",
        len(results)
    )

    return results