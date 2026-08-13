#Responsible for receiving file events, printing/logging them, later creating event objects

from watchdog.events import FileSystemEventHandler #react for file modify, create, delete
from watchdog.observers import Observer #watch a folder continously
import time
from config import *
from app.utils.file_types import FILE_TYPES
import os
# from app.celery_app.tasks.process_file import route_file
from app.kafka_producer.producer import publish_file_event
from app.utils.hash import calculate_file_hash

# from app.storage.lancedb_store import is_duplicate

# print("WATCHED_FOLDER =", WATCHED_FOLDER)
# print("Exists =", os.path.exists(WATCHED_FOLDER))

class SecondBrainHandler(FileSystemEventHandler):
	def on_created(self, event):
		if not event.is_directory: 
			path=event.src_path

			#Wait until file copy finishes
			last_size=-1
			while True:
				
				if not os.path.exists(path):
					print("File no longer exists.")
					return
				current_size=os.path.getsize(path)
				if current_size==last_size:
					break
				last_size=current_size
				time.sleep(1)

			print(f"A new file is created: {path}")
			print(f"Final Size: {os.path.getsize(path)} bytes")

			# Skip zero-byte files (file was created but nothing was written)
			if os.path.getsize(path) == 0:
				print(f"Skipping zero-byte file: {path}")
				return

			file_hash = calculate_file_hash(path)
			print(file_hash)
			# Check duplicate
			# if is_duplicate(file_hash):
			# 	print("Duplicate file detected. Skipping...")
			# 	return

			extension = os.path.splitext(path)[1].lower()

			file_type = FILE_TYPES.get(extension)
			if file_type is None:
				print(f"Unsupported file type: {extension}")
				return

			publish_file_event({
				"path":path,
				"file_type":file_type,
				"file_hash": file_hash
				})
			print("Published event to Kafka.")

	def on_modified(self, event):
		if not event.is_directory:
			print(f"Modified: {event.src_path}")

	def on_deleted(self, event):
		if not event.is_directory:
			path=event.src_path
			print(f"File deleted: {path}")

			publish_file_event({
				"path":path,
				"event_type":"deleted"
				})		
			print("Delete event published to Kafka")

	def on_moved(self, event):
		if not event.is_directory:
			print(f"File moved from {event.src_path} to {event.dest_path}")

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

