from app.celery_app.celery import app
import logging
import os
import uuid

from app.extractors.router import extract_file
from app.storage.lancedb_store import save_file_hash, is_file_processed, store_chunk, store_image_chunk, store_face_record, deduplicate_and_store_faces, delete_source_records
from app.embeddings.embedding_router import generate_embedding
from app.chunker.chunker import chunk_text
from app.embeddings.image_embedding import embed_image
from app.embeddings.face_embedding import detect_and_embed_faces

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)

def route_file(self, event):
    path = event["path"]
    event_type = event.get("event_type", "created")
    if event_type == "deleted":
        delete_source_records(path)
        return {"status": "deleted", "path": path}
    file_type = event["file_type"]
    file_hash = event["file_hash"]

    # Skip duplicate files
    if is_file_processed(file_hash):
        logger.info(f"Duplicate file skipped: {path}")
        return {"status": "duplicate", "path": path}

    logger.info(f"Started processing: {path}")
    delete_source_records(path)

    filename = os.path.basename(path)
    extracted_data = extract_file(path)

    # Check status for structured extraction returns
    if isinstance(extracted_data, dict):
        status = extracted_data.get("status", "SUCCESS")
        if status != "SUCCESS":
            logger.warning(f"File processing failed for {path}: status={status}, error={extracted_data.get('error')}")
            raise RuntimeError(extracted_data.get("error") or f"Extraction failed with status {status}")

    chunks = []

    #------------------------
    #Image file
    #------------------------
    if file_type=='image':
        text_content=""

        if isinstance(extracted_data, dict):
            text_content=extracted_data.get("text","")
        else:
            text_content=str(extracted_data)
        text_chunks=chunk_text(text_content)
        
        #If image has OCR/face information, create a searchable chunk
        for c in text_chunks:
            chunks.append(f"File: {filename}\n{c}")

        #if there is no OCR text, still keep one image record so the actual image embedding can be stored later.
        if not chunks:
            chunks.append(f"File: {filename}\nImage")
        logger.info(f"Created {len(chunks)} image chunk(s) for {filename}")        

        #generate 512-D CLIP Image embedding
        image_embedding=embed_image(path)
 
        chunk_id=str(uuid.uuid4())
        store_image_chunk(
                chunk_id=chunk_id,
                path=path,
                file_type=file_type,
                text="\n".join(chunks),
                image_embedding=image_embedding
        )

        # Store image text chunks in main text vector table for unified RAG text search
        for chunk in chunks:
            c_id = str(uuid.uuid4())
            embedding = generate_embedding(file_type, chunk)
            store_chunk(
                chunk_id=c_id,
                path=path,
                file_type=file_type,
                text=chunk,
                embedding=embedding
            )

        #Extract face embeddings and store in face_embeddings table
        try:
            detected_faces = detect_and_embed_faces(path)
            stored_recs = deduplicate_and_store_faces(path, detected_faces)
            if detected_faces:
                logger.info(f"Stored/deduplicated {len(stored_recs)} face embedding(s) for {filename}")
        except Exception as e:
            logger.warning(f"Failed to extract face embeddings for {filename}: {e}")

    #------------------------        
    #PDF/Other paged documents        
    #------------------------
    else:        



        # Case 1: PDF extraction returned structured dict with page dictionaries or raw page list
        pages = []
        if isinstance(extracted_data, dict) and extracted_data.get("pages"):
            pages = extracted_data["pages"]
        elif isinstance(extracted_data, list):
            pages = extracted_data

        if pages:
            for page_info in pages:
                if isinstance(page_info, dict):
                    page_num = page_info.get("page", 1)
                    page_text = page_info.get("text", "")

                    page_chunks = chunk_text(page_text)
                    for c in page_chunks:
                        header = f"File: {filename} | Page: {page_num}"
                        chunks.append(f"{header}\n{c}")

        # Case 2: Standard structured dict or string return
        else:
            text_content = ""
            if isinstance(extracted_data, dict):
                text_content = extracted_data.get("transcript") or extracted_data.get("text", "")
            else:
                text_content = str(extracted_data)

            text_chunks = chunk_text(text_content)
            for c in text_chunks:
                chunks.append(f"File: {filename}\n{c}")

        # If extraction yielded no text chunks, do NOT create dummy chunks or LanceDB records
        if not chunks:
            logger.info(f"No text chunks created for {filename}")
            save_file_hash(file_hash=file_hash, path=path)
            return {"status": "NO_TEXT_EXTRACTED", "path": path, "total_chunks": 0}

        logger.info(f"Created {len(chunks)} chunks for {filename}")

        # Generate vector embeddings and save to LanceDB with explicit sequence indices
        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_id = f"{file_hash}_chunk_{chunk_idx}"
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

    # Build and save source_catalog entry using local Ollama (ask_llama)
    try:
        from app.query.source_catalog import build_entry, save_entry
        from app.llm.ollama_client import ask_llama
        sample_text = "\n".join(chunks)[:2500] if chunks else ""
        entry = build_entry(path, sample_text, ask_llama, file_type)
        save_entry(entry)
        logger.info(f"Source catalog entry saved for {path}")
    except Exception as exc:
        logger.warning(f"Failed to save source catalog entry for {path}: {exc}")

    logger.info(f"Finished processing: {path}")

    return {
        "path": path,
        "file_type": file_type,
        "total_chunks": len(chunks),
        "status": "processed",
    }