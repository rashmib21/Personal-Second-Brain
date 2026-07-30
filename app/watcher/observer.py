#Responsible for Creating the Watchdog observer, starting it, stopping it  

import time
from watchdog.observers import Observer
from config import WATCHED_FOLDER
from app.watcher.handler import SecondBrainHandler

observer=Observer()
handler=SecondBrainHandler()

observer.schedule(handler, WATCHED_FOLDER, recursive=True)
observer.start()

print(f"Watching: {WATCHED_FOLDER}")

try: 
	while True:
		time.sleep(1)
except KeyboardInterrupt:
	observer.stop()

observer.join()			