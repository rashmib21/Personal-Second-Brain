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
    summarize the audio.mpeg -> audio.mpeg
    summarize the audio MLKDream -> MLKDream.mp3
    """
    if df is None or df.empty or "path" not in df.columns:
        return None

    question_lower = question.lower()

    # Common English stop words and command filler words
    stop_words = {
        "the", "a", "an", "of", "in", "from", "to", "and", "or", "is", "for",
        "with", "on", "at", "by", "this", "that", "my", "please",
        "summarize", "summarise", "summary", "chapter", "section", "part", "file", "show", "get"
    }

    # Media type category terms
    media_terms = {
        "audio", "video", "image", "picture", "photo", "document", "pdf",
        "doc", "docx", "txt", "text", "mp3", "mp4", "mpeg", "wav", "aac", "flac"
    }

    best_path = None
    best_score = -1

    for path in df["path"].unique():
        if not path or not isinstance(path, str):
            continue

        filename = os.path.basename(path)
        filename_lower = filename.lower()
        stem = os.path.splitext(filename)[0].lower()
        clean_name = clean_filename(filename)

        score = 0

        # 1. Highest priority: Exact full filename (e.g. "audio.mpeg") in question -> 1000 pts
        if filename_lower in question_lower:
            score += 1000

        # 2. High priority: Exact non-generic stem (e.g. "mlkdream") in question -> 500 pts
        elif stem and stem not in stop_words and stem not in media_terms and stem in question_lower:
            score += 500

        # 3. Medium priority: Token word matches
        query_words = re.findall(r"\b[a-z0-9]+\b", question_lower)
        filename_words = re.findall(r"\b[a-z0-9]+\b", clean_name)

        for word in query_words:
            if word in stop_words or len(word) < 2:
                continue
            if word in filename_words or (stem and word in stem):
                if word in media_terms:
                    score += 5   # lower score for generic category words
                else:
                    score += 20  # higher score for specific title words

        if score > best_score and score > 0:
            best_score = score
            best_path = path

    return best_path


def is_toc_chunk(text):
    """
    Checks whether a chunk looks like Table of Contents, copyright, author credits,
    or structural metadata rather than body content.
    """
    if not text:
        return True

    text_lower = text.lower().strip()
    if not text_lower:
        return True

    # Common TOC and structural metadata terms
    metadata_terms = [
        "table of contents", "brief contents", "contents in detail",
        "about this book", "you may also like", "goalkicker.com",
        "free programming books", "contributors", "acknowledgements",
        "acknowledgments", "license", "copyright", "disclaimer"
    ]
    for term in metadata_terms:
        if term in text_lower:
            return True

    if text_lower.startswith("contents"):
        return True

    # Dotted TOC lines (e.g. Chapter 1 ......... 12)
    if re.search(r"\.{3,}", text):
        return True

    # Multiple distinct chapter headings in a single chunk (e.g. Chapter 1, Chapter 2, Chapter 3...)
    chapter_matches = re.findall(r"\bchapter\s*\d+\b", text_lower)
    if len(set(chapter_matches)) > 2:
        return True

    # Bibliography or Index sections
    if "bibliography" in text_lower or "\nindex" in text_lower or text_lower.startswith("index"):
        return True

    # Check for author/chapter listing range patterns (e.g. Chapters 1-62: Various authors)
    if "chapter" in text_lower and re.search(r"\bchapters\s*\d+\s*[-–to]\s*\d+", text_lower):
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

#------summary of whole book------------
def search_for_book_summary(question, max_chunks=None):
    #Retrieve all useful chunks from a document for whole book summarization
    #get lancedb table
    table=get_table()
    if table is None:
        return []
    df=table.to_pandas()
    
    if df.empty:
        return []

    #find the document with user's question
    source_path=find_document(question, df)

    if source_path is None:
        return []

    #Keep only requested document
    document_df=df[
        df['path']==source_path]    


    if document_df.empty:
        return []

    #Sort chunks
    if "chunk_index" in document_df.columns:
        document_df=document_df.sort_values("chunk_index")

    #Convert rows
    rows=list(document_df.itertuples(index=False))

    #Collect useful chunks
    results=[]
    for row in rows:
        text=str(row.text)

        #Ignore empty chunks
        if not text.strip():
            continue

        #Ignore TOC/index/bibliography
        if is_toc_chunk(text):
            continue 

        results.append({
                "chunk_id":row.chunk_id,
                "path":row.path,
                "file_type":row.file_type,
                "text":text
            })

        #Safety limit
        if(max_chunks is not None and len(results)>=max_chunks):
            break

    return results


def get_clean_document_summary_context(source_path, max_chars=25000):
    """
    Retrieves clean, non-TOC body content for a document.
    If total body text fits inside max_chars, returns full text.
    If total body text exceeds max_chars, performs uniform sampling across
    the document's sequence of body chunks to represent beginning, middle, and end.

    Returns tuple: (context_str: str, total_chunks: int, sampled_count: int)
    """
    table = get_table()
    if table is None:
        return "", 0, 0

    df = table.to_pandas()
    if df.empty or "path" not in df.columns:
        return "", 0, 0

    target_base = os.path.basename(source_path).lower()
    source_path_lower = str(source_path).lower()

    # Filter rows for target source
    matching_mask = (df["path"].str.lower() == source_path_lower) | (
        df["path"].apply(lambda p: os.path.basename(str(p)).lower()) == target_base
    )
    document_df = df[matching_mask].copy()

    if document_df.empty:
        return "", 0, 0

    def extract_chunk_idx(row):
        cid = str(row.get("chunk_id", ""))
        match = re.search(r"_chunk_(\d+)", cid)
        if match:
            return int(match.group(1))
        return 0

    document_df["_sort_idx"] = document_df.apply(extract_chunk_idx, axis=1)
    document_df = document_df.sort_values("_sort_idx")

    rows = list(document_df.itertuples(index=False))
    total_chunks = len(rows)

    # Filter out TOC / metadata / author list chunks
    body_chunks = []
    for row in rows:
        text = str(row.text or "").strip()
        if not text:
            continue
        if is_toc_chunk(text):
            continue
        body_chunks.append(text)

    # Fallback: if filtering removed all chunks, use original non-empty chunks
    if not body_chunks:
        body_chunks = [str(r.text).strip() for r in rows if str(r.text or "").strip()]

    if not body_chunks:
        return "", 0, 0

    total_body_text = "\n\n".join(body_chunks)

    # Case 1: Total body text fits within max_chars limit
    if len(total_body_text) <= max_chars:
        return total_body_text, total_chunks, len(body_chunks)

    # Case 2: Uniform sampling across document sequence
    num_body = len(body_chunks)
    avg_chunk_len = sum(len(c) for c in body_chunks) / num_body
    target_count = max(5, int(max_chars / max(1, avg_chunk_len)))
    target_count = min(target_count, num_body)

    indices = [
        int(i * (num_body - 1) / (target_count - 1)) for i in range(target_count)
    ] if target_count > 1 else [0]
    indices = sorted(list(set(indices)))

    sampled_chunks = [body_chunks[idx] for idx in indices]
    context_str = "\n\n--- SECTION BREAK ---\n\n".join(sampled_chunks)

    return context_str, total_chunks, len(sampled_chunks)

