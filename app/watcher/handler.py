#Responsible for receiving file events, printing/logging them, later creating event objects

from watchdog.events import FileSystemEventHandler #react for file modify, create, delete
from watchdog.observers import Observer #watch a folder continously
import time
from config import *
import os
from app.celery_app.tasks.process_file import process_file

# print("WATCHED_FOLDER =", WATCHED_FOLDER)
# print("Exists =", os.path.exists(WATCHED_FOLDER))

class SecondBrainHandler(FileSystemEventHandler):
	def on_created(self, event):
		if not event.is_directory: 
			print(f"A new file is created: {event.src_path}")
			process_file.delay(event.src_path)
			print("Task sent to Celery.")

	def on_modified(self, event):
		if not event.is_directory:
			print(f"Modified: {event.src_path}")

observer=Observer() #security guard, watching the folder

handler=SecondBrainHandler() #handler reacting on that event

#observer watch this folder and something happens, call the handler
observer.schedule(handler, WATCHED_FOLDER,recursive=True)


observer.start() #start watching

try:
	while True:
		time.sleep(1)

except KeyboardInterrupt:
	observer.stop()

observer.join()	#wait here until the observer has completely stopped




