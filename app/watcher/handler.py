#Responsible for receiving file events, printing/logging them, later creating event objects

from watchdog.events import FileSystemEventHandler #react for file modify, create, delete
from watchdog.observers import Observer #watch a folder continously
import time
from config import *
import os
from app.celery_app.tasks.process_file import route_file
from app.utils.hash import calculate_file_hash
# from app.storage.lancedb_store import is_duplicate

# print("WATCHED_FOLDER =", WATCHED_FOLDER)
# print("Exists =", os.path.exists(WATCHED_FOLDER))

class SecondBrainHandler(FileSystemEventHandler):
	def on_created(self, event):
		if not event.is_directory: 
			path=event.src_path
			print(f"A new file is created: {path}")
			file_hash = calculate_file_hash(path)
			print(file_hash)
			# Check duplicate
			# if is_duplicate(file_hash):
			# 	print("Duplicate file detected. Skipping...")
			# 	return
			file_type = os.path.splitext(path)[1].lower().lstrip(".")
			route_file.delay({
				"path":path,
				"file_type":file_type,
				"file_hash": file_hash
				})
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

