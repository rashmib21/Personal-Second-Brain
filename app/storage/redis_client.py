#it is mainly responsible for redisearch index that makes semantic (vector) search possible
import redis

from redis.commands.search.field import VectorField, TextField #(the paramaters are path, file type, embedding-vector - the first two is text and third one is vector)
#redis perform semantic search on vector

from redis.commands.search.indexDefinition import(IndexDefinition, IndexType)
#only build index for keys beginning with chunks, IndexType is used for storing HASH value in key pair -path, file_type, text, embeddings etc

#Connect to redis
r=redis.Redis(
	host=REDIS_HOST,
	port=REDIS_PORT) #this creates connection with redis server