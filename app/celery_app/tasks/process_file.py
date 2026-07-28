from app.celery_app.celery import app
import logging

from app.extractors.pdf import extract_pdf
from app.extractors.document import extract_document
from app.extractors.text import extract_text
from app.extractors.image import extract_image
from app.extractors.audio import extract_audio
from app.extractors.video import extract_video

from app.chunker.chunker import chunk_text


logger = logging.getLogger(__name__)


# Maps each file type to its extractor
TASKS = {
    "pdf": extract_pdf,
    "document": extract_document,
    "text": extract_text,
    "image": extract_image,
    "audio": extract_audio,
    "video": extract_video,
}


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def route_file(self, event):

    # Get information from the Kafka event
    path = event["path"]
    file_type = event["file_type"]

    logger.info(f"Started processing : {path}")

    # Find the correct extractor
    processor = TASKS.get(file_type)

    if processor is None:
        raise ValueError(f"Unsupported file type : {file_type}")

    # Extract content
    extracted_data = processor(path)

    # Video returns transcript + keyframes
    if file_type == "video":

        transcript = extracted_data["transcript"]
        keyframes = extracted_data["keyframes"]

        chunks = chunk_text(transcript)

        logger.info(f"Extracted {len(keyframes)} keyframes")

    # Image returns OCR text
    elif file_type == "image":

        chunks = chunk_text(extracted_data)

    # Audio returns transcript
    elif file_type == "audio":

        chunks = chunk_text(extracted_data)

    # PDF / Document / Text return plain text
    else:

        chunks = chunk_text(extracted_data)

    logger.info(f"Created {len(chunks)} chunks")

    # Print chunks temporarily
    for i, chunk in enumerate(chunks, start=1):
        logger.info(f"Chunk {i}:\n{chunk}\n")

    logger.info(f"Finished processing : {path}")

    return {
        "path": path,
        "file_type": file_type,
        "total_chunks": len(chunks),
        "status": "processed",
    }