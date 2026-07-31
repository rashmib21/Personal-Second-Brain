from app.celery_app.celery import app
import logging

# from app.extractors.pdf import extract_pdf
# from app.extractors.document import extract_document
# from app.extractors.text import extract_text
# from app.extractors.image import extract_image
# from app.extractors.audio import extract_audio
# from app.extractors.video import extract_video


from app.extractors.router import extract_file
from app.storage.redis_store import save_file_hash
from app.embeddings.embedding_router import generate_embedding
from app.storage.redis_store import store_chunk
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

    logger.info(f"Started processing : {path}")

    extracted_data=extract_file(path)

    # Video returns transcript + keyframes
    if isinstance(extracted_data, dict):

        transcript = extracted_data["transcript"]
        keyframes = extracted_data["keyframes"]

        chunks = chunk_text(transcript)

        logger.info(f"Extracted {len(keyframes)} keyframes")

    else:

        chunks = chunk_text(extracted_data)

    logger.info(f"Created {len(chunks)} chunks")

    #Generate embeddings and store in Redis
    for chunk in chunks:

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