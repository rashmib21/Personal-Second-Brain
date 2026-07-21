from celery_app.celery import app
import logging
import time

logger=logging.getLogger(__name__)

#retry_backoff=celery waits for sometime before retry after failure
@app.task(bind=True, autoretry_for=(Exception,),retry_backoff=True, retry_kwargs={"max_retries":3})


def process_file(self, path):
	logger.info(f"Started Processing {path}")

	#long running task
	time.sleep(30)
	logger.info(f"Finished Processing {path}")
	
	return {"path":path, "status":"processed"}
