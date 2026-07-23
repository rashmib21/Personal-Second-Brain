from app.celery_app.celery import app
import logging
import time
import os

from app.extractors.document import extract_text
from app.chunker.chunker import chunk_text


logger=logging.getLogger(__name__)

#retry_backoff=celery waits for sometime before retry after failure
@app.task(bind=True, autoretry_for=(Exception,),retry_backoff=True, retry_kwargs={"max_retries":3})


def process_file(self, path):

	file_name=os.path.basename(path)
	print(f"*****{file_name}*****")

	logger.info(f"Started Processing {path}")

	#Step 1: Extract text from the document
	text=extract_text(path)

	logger.info("Text extracted successfully")

	#Step 2: Split text into chunks
	chunks=chunk_text(text)

	logger.info(f"Created {len(chunks)} chunks")

	#print chunks temporarily
	for i, chunk in enumerate(chunks, start=1):
		logger.info(f"Chunk {i}:\n{chunk}\n")

	#long running task
	# time.sleep(30)
	logger.info(f"Finished Processing {path}")

	return {
		"path":path, 
		"total_chunks":len(chunks), 
		"status":"processed"
	}
