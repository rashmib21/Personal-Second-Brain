#this file is created to store embeddings permanently

import redis #Connect to redis database
import numpy as np #convert the embedding (python list) into binary bytes because redis expects vectors in byte format
from config import REDIS_HOST, REDIS_PORT

r=redis.Redis(
	host=REDIS_HOST,
	port=REDIS_PORT,
	decode_responses=False #normally redis converts bytes into string, but embeddings are binary data, not text
	)

def write_chunk(chunk_id, path, file_type, embedding):
	
	#Convert python list into numpy array
	vector=np.array(embedding,dtype=np.float32)

	#Convert numpy array into bytes 
	vector_bytes=vector.tobytes() 

	#Store the chunk inside Redis
	r.hset(
		f"chunk: {chunk_id}",
		mapping={
		"path":path,
		"file_type":file_type,
		"embedding":vector_bytes
		}
	)

	print(f"Chunk: {chunk_id} stored successfully!")

if __name__=='__main__':
	write_chunk(
		chunk_id=1,
		path='watched_folder/Rashmi_Barethiya_19-05',
		file_type="pdf",
		embedding=[0.1]*384
		)