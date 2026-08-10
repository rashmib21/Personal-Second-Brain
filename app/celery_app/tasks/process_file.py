from app.celery_app.celery import app
import logging
import os

# from app.extractors.pdf import extract_pdf
# from app.extractors.document import extract_document
# from app.extractors.text import extract_text
# from app.extractors.image import extract_image
# from app.extractors.audio import extract_audio
# from app.extractors.video import extract_video


from app.extractors.router import extract_file
from app.storage.lancedb_store import save_file_hash, is_file_processed
from app.embeddings.embedding_router import generate_embedding
from app.storage.lancedb_store import store_chunk
import uuid

from app.chunker.chunker import chunk_text


logger = logging.getLogger(__name__)



@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def route_file(self, event):

    print("EVENT RECEIVED:", event)
    print("EVENT TYPE:", type(event))

    # Get information from the Kafka event
    path = event["path"]
    file_type = event["file_type"]
    file_hash = event["file_hash"]

    if is_file_processed(file_hash):
        logger.info(f"Duplicate file skipped: {path}")

        return {
            "status":"duplicate",
            "path":path
        }    

    logger.info(f"Started processing : {path}")

    extracted_data=extract_file(path)

    # Video returns transcript + keyframes
    if isinstance(extracted_data, dict):
        #Video
        if "keyframes" in extracted_data:
            transcript = extracted_data["transcript"]
            keyframes = extracted_data["keyframes"]
            chunks = chunk_text(transcript)
            logger.info(f"Extracted {len(keyframes)} keyframes")

        #Audio
        elif "transcript" in extracted_data:
            transcript = extracted_data['transcript']
            chunks = chunk_text(transcript)

        else:
            raise ValueError("Unknown extracted data format")        

    else:

        chunks = chunk_text(extracted_data)

    filename = os.path.basename(path)

    # If transcript/text is empty, still create at least one chunk with filename metadata
    if not chunks:
        chunks = [f"File: {filename}"]
    else:
        # Prepend filename context to every chunk so vector search matches file names and content
        chunks = [f"File: {filename}\n{chunk}" for chunk in chunks]

    logger.info(f"Created {len(chunks)} chunks")

    print(chunks)
    print(len(chunks))
    #Generate embeddings and store in lancedb
    for chunk in chunks:
        print("Chunk =>", chunk[:100])

        #Unique ID for every chunk
        chunk_id=str(uuid.uuid4())

        #Convert text into vector
        embedding=generate_embedding(file_type, chunk)

        #Store metadata and vector
        store_chunk(
            chunk_id=chunk_id,
            path=path,
            file_type=file_type,
            text=chunk,
            embedding=embedding
        )
        print("Saved to LanceDB")

    # Mark file as processed

    save_file_hash(
        file_hash=file_hash,
        path=path
    )

    logger.info(f"Finished processing: {path}")
        
    return {
        "path":path,
        "file_type":file_type,
        "total_chunks":len(chunks),
        "status":"processed",
        }    