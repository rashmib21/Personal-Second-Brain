import re
import os

from app.storage.lancedb_store import get_table


def get_chapter_number(question):
    """
    Gets chapter number from user query.

    Example:
    summarize chapter 1 of sql professionals
    -> 1
    """

    question = question.lower()

    match = re.search(
        r"chapter\s+(\d+)",
        question
    )

    if match:
        return int(match.group(1))

    return None


def clean_filename(filename):
    """
    Converts filename into simple words.

    Example:

    SQLNotesForProfessionals.pdf

    becomes:

    sql notes for professionals
    """

    filename = os.path.splitext(filename)[0]

    # Split CamelCase
    filename = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        filename
    )

    filename = filename.replace("_", " ")
    filename = filename.replace("-", " ")

    filename = filename.lower()

    return filename


def find_document(question, df):
    """
    Finds the document that best matches the user's query.

    Example:

    sql professionals
    ->
    SQLNotesForProfessionals.pdf

    linux
    ->
    linux.pdf
    """

    question = question.lower()

    query_words = re.findall(
        r"\b[a-z0-9]+\b",
        question
    )

    ignored_words = [
        "summarize",
        "summarise",
        "summary",
        "chapter",
        "section",
        "part",
        "of",
        "the",
        "from",
        "in",
        "my",
        "please"
    ]

    useful_words = []

    for word in query_words:

        if word in ignored_words:
            continue

        if len(word) < 2:
            continue

        useful_words.append(word)

    best_path = None
    best_matches = 0

    for path in df["path"].unique():

        filename = os.path.basename(path)

        clean_name = clean_filename(filename)

        filename_words = re.findall(
            r"\b[a-z0-9]+\b",
            clean_name
        )

        matches = 0

        for word in useful_words:

            if word in filename_words:
                matches = matches + 1

        if matches > best_matches:

            best_matches = matches
            best_path = path

    return best_path


def is_toc_chunk(text):
    """
    Checks whether a chunk looks like
    Table of Contents or other structural material.
    """

    text_lower = text.lower()

    # Table of contents
    if "table of contents" in text_lower:
        return True

    # Brief contents
    if "brief contents" in text_lower:
        return True

    # Detailed contents
    if "contents in detail" in text_lower:
        return True

    # Contents heading
    if text_lower.strip().startswith("contents"):
        return True

    # Dotted TOC lines
    #
    # Example:
    # Chapter 1: Getting Started ........ 1

    if re.search(r"\.{3,}", text):
        return True

    # Multiple chapter headings usually
    # means this is a TOC.
    chapter_matches = re.findall(
        r"\bchapter\s*\d+\b",
        text_lower
    )

    if len(chapter_matches) > 1:
        return True

    # Bibliography
    if "bibliography" in text_lower:
        return True

    # Index
    if "\nindex" in text_lower:
        return True

    return False


def is_chapter_heading(text, chapter_number):
    """
    Checks whether a chunk contains
    the requested chapter heading.

    Supports:

    Chapter 1: Getting started with SQL

    Chapter1: Getting started with SQL

    1
    THE BIG PICTURE
    """

    text_lower = text.lower()

    # --------------------------------
    # Format 1
    #
    # Chapter 1
    # Chapter 1:
    # Chapter 1: Title
    # --------------------------------

    pattern = rf"\bchapter\s*{chapter_number}\b"

    if re.search(
        pattern,
        text_lower
    ):
        return True

    # --------------------------------
    # Format 2
    #
    # Linux book can have:
    #
    # 1
    # THE BIG PICTURE
    # --------------------------------

    lines = text.splitlines()

    for i in range(len(lines)):

        line = lines[i].strip()

        if line == str(chapter_number):

            if i + 1 < len(lines):

                next_line = lines[i + 1].strip()

                if next_line:

                    return True

    return False


def is_next_chapter(text, chapter_number):
    """
    Checks whether the next chapter has started.
    """

    next_chapter = chapter_number + 1

    pattern = rf"\bchapter\s*{next_chapter}\b"

    if re.search(
        pattern,
        text,
        re.IGNORECASE
    ):
        return True

    return False


def search_for_summary(question, max_chunks=30):
    """
    Finds the actual chunks belonging
    to the requested chapter.

    This does NOT use vector search.

    It reads the document chunks from
    LanceDB and follows their document order.
    """

    # -----------------------------------
    # 1. Get LanceDB table
    # -----------------------------------

    table = get_table()

    if table is None:
        return []

    df = table.to_pandas()

    if df.empty:
        return []

    # -----------------------------------
    # 2. Get chapter number
    # -----------------------------------

    chapter_number = get_chapter_number(question)

    if chapter_number is None:
        return []

    # -----------------------------------
    # 3. Find document
    # -----------------------------------

    source_path = find_document(
        question,
        df
    )

    if source_path is None:
        # print("SUMMARY DOCUMENT NOT FOUND")
        return []

    # print(
    #     "SUMMARY DOCUMENT:",
    #     source_path
    # )

    # print(
    #     "SUMMARY CHAPTER:",
    #     chapter_number
    # )

    # -----------------------------------
    # 4. Keep only requested document
    # -----------------------------------

    document_df = df[
        df["path"] == source_path
    ]

    if document_df.empty:
        return []

    # -----------------------------------
    # 5. Convert rows to list
    # -----------------------------------

    rows = list(
        document_df.itertuples(index=False)
    )

    # -----------------------------------
    # 6. Find actual chapter start
    # -----------------------------------

    start_position = None

    for position in range(len(rows)):

        row = rows[position]

        text = str(row.text)

        # Ignore TOC chunks
        if is_toc_chunk(text):
            continue

        # Check for chapter heading
        if is_chapter_heading(
            text,
            chapter_number
        ):

            start_position = position

            # print(
            #     "CHAPTER START FOUND:"
            # )

            # print(
            #     text[:500]
            # )

            break

    # -----------------------------------
    # 7. Chapter not found
    # -----------------------------------

    if start_position is None:

        # print(
        #     "CHAPTER START NOT FOUND"
        # )

        return []

    # -----------------------------------
    # 8. Collect chapter chunks
    # -----------------------------------

    results = []

    for position in range(
        start_position,
        len(rows)
    ):

        row = rows[position]

        text = str(row.text)

        # --------------------------------
        # Stop at next chapter
        # --------------------------------

        if position > start_position:

            if is_next_chapter(
                text,
                chapter_number
            ):

                # print(
                #     "NEXT CHAPTER FOUND"
                # )

                break

        # --------------------------------
        # Add chunk
        # --------------------------------

        results.append({
            "chunk_id": row.chunk_id,
            "path": row.path,
            "file_type": row.file_type,
            "text": text
        })

        # print(
        #     "SUMMARY CHUNK:",
        #     position,
        #     text[:200]
        # )

        # --------------------------------
        # Safety limit
        # --------------------------------

        if len(results) >= max_chunks:

            break

    # print(
    #     "TOTAL SUMMARY CHUNKS:",
    #     len(results)
    # )

    return results