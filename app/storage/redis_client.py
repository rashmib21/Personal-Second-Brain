#it is mainly responsible for redisearch index that makes semantic (vector) search possible
import redis
import numpy

from redis.commands.search.field import VectorField, TextField #(the paramaters are path, file type, embedding-vector - the first two is text and third one is vector)
#redis perform semantic search on vector

from redis.commands.search.index_definition import(IndexDefinition, IndexType)
#only build index for keys beginning with chunks, IndexType is used for storing HASH value in key pair -path, file_type, text, embeddings etc

from config import REDIS_HOST, REDIS_PORT

#Connect to redis
r=redis.Redis(
	host=REDIS_HOST,
	port=REDIS_PORT) #this creates connection python with redis server


#Initialize a function for creating index
def create_index(dim=384): #every embedding model return vector of fixed length
	schema=( #describe what every documents look like
		TextField('path'),
		TextField('file_type'),
		VectorField('embeddings', #Name of field
			'HNSW', #HNSW make fast searching- nearest to destination,
			{
				"TYPE":"FLOAT32", #Vector data type
				"DIM":dim, #Number of values in each vector
				"DISTANCE_METRIC":"COSINE" #Comapre vectors using cosine similarity
			},
		),
	)  
	

	#ft() tells the Redis client: I want to use the RediSearch module.
	#Create an index named "idx:files"
	try:
		r.ft("idx:files").info()
		print("Index already exists.")
	
	except :
		r.ft("idx:files").create_index(

		#Fields to index
		schema,

		#Tell Redis which keys beginning with "chunks:"
		definition=IndexDefinition(

			#Only index keys beginning with chunks
			prefix=['chunk:'],

			#Those keys are stored as Redis HASHes
			index_type=IndexType.HASH,
			),
		)
	print("RediSearch index created successfully!")

def store_chunk(chunk_id, path, file_type, text, embedding):
	#Store one chunk
	key=f"chunk: {chunk_id}"

	#numpy is used for converting the python list into bytes because redisearch only understand raw bytes and numoy knows how to convert
	embedding=np.array(embedding, dtype=np.float32).tobytes()

	r.hset(key, mapping={
		"path":path,
		"file_type":file_type,
		"text":text,
		"embeddings":embedding,
		},)

	print(f"Stored {key} in Redis")

if __name__=='__main__':
	create_index()			

