from app.celery_app.celery import app
import logging
import os
import uuid

from app.extractors.router import extract_file
from app.storage.lancedb_store import save_file_hash, is_file_processed, store_chunk
from app.embeddings.embedding_router import generate_embedding
from app.chunker.chunker import chunk_text

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def route_file(self, event):
    path = event["path"]
    file_type = event["file_type"]
    file_hash = event["file_hash"]

    # Skip duplicate files
    if is_file_processed(file_hash):
        logger.info(f"Duplicate file skipped: {path}")
        return {"status": "duplicate", "path": path}

    logger.info(f"Started processing: {path}")

    filename = os.path.basename(path)
    extracted_data = extract_file(path)

    chunks = []

    # Case 1: PDF extraction returned a list of page dictionaries
    if isinstance(extracted_data, list):
        for page_info in extracted_data:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")

            page_chunks = chunk_text(page_text)
            for c in page_chunks:
                header = f"File: {filename} | Page: {page_num}"
                chunks.append(f"{header}\n{c}")

    # Case 2: Audio/Video extraction returned a dictionary with transcript
    elif isinstance(extracted_data, dict):
        transcript = extracted_data.get("transcript", "")
        text_chunks = chunk_text(transcript)
        for c in text_chunks:
            chunks.append(f"File: {filename}\n{c}")

    # Case 3: Plain text / Word / Spreadsheet / standard string output
    else:
        text_chunks = chunk_text(str(extracted_data))
        for c in text_chunks:
            chunks.append(f"File: {filename}\n{c}")

    # Ensure at least one chunk exists if extraction returned empty text
    if not chunks:
        chunks = [f"File: {filename}"]

    logger.info(f"Created {len(chunks)} chunks for {filename}")

    # Generate vector embeddings and save to LanceDB
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        embedding = generate_embedding(file_type, chunk)

        store_chunk(
            chunk_id=chunk_id,
            path=path,
            file_type=file_type,
            text=chunk,
            embedding=embedding
        )

    # Save hash to avoid re-processing same file
    save_file_hash(file_hash=file_hash, path=path)

    logger.info(f"Finished processing: {path}")

    return {
        "path": path,
        "file_type": file_type,
        "total_chunks": len(chunks),
        "status": "processed",
    }