#Responsible for receiving file events, printing/logging them, later creating event objects

from watchdog.events import FileSystemEventHandler #react for file modify, create, delete
from watchdog.observers import Observer #watch a folder continously
import time

class SecondBrainHandler(FileSystemEventHandler):
	def on_created(self, event):
		if not event.is_directory: 
			print("A new file is created: {event.src_path}")

			
