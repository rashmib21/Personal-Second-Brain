import os 
from dotenv import load_dotenv

#Load variables from .env file
load_dotenv()

WATCHED_FOLDER=os.getenv("WATCHED_FOLDER","./watched_files")
LOG_LEVEL=os.getenv("LOG_LEVEL","INFO")

KAFKA_BOOTSTRAP_SERVERS=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

REDIS_HOST=os.getenv("REDIS_HOST","localhost")
REDIS_PORT=int(os.getenv("REDIS_PORT",6379))

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY","")
