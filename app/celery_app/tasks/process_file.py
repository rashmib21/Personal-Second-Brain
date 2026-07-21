from celery_app.celery import app

#retry_backoff=celery waits for sometime before retry after failure
@app.task(bind=True, autoretry_for=(Exception,),retry_backoff=True, retry_kwargs={"max_retries":3})

