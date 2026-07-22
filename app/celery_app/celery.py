from celery import Celery
from config import REDIS_HOST, REDIS_PORT

app=Celery(
	"second-brain",
	broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
	backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/1",
	)

app.conf.task_acks_late=True #the broker marks as Done after the tasks finishes
app.conf.task_reject_on_worker_lost=True  #if worker fails tasks goes back to the redis queue


# import app.celery_app.tasks.process_file

app.conf.imports=(
	"app.celery_app.tasks.process_file",
	)
