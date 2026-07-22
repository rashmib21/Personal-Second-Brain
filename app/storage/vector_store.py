#this file is created to store embeddings permanently

import redis #Connect to redis database
import numpy as np #convert the embedding (python list) into binary bytes because redis expects vectors in byte format
from config import REDIS_HOST, REDIS_PORT

r=redis.Redis(
	host=REDIS_HOST,
	port=REDIS_PORT,
	decode_response=False #normally redis converts bytes into string, but embeddings are binary data, not text
	)