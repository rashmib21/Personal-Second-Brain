import numpy as np
from redis.commands.search.query import Query

from app.storage.redis_store import r
from app.embeddings.embedding_router import generate_embedding

MAX_ALLOWED_DISTANCE = 0.5

def search(question, how_many_results=5):
	#convert user query into embeddings
	question_vector=generate_embedding(
		"text",
		question
	)

	#Convert embedding list to bytes
	query_vector_bytes=np.array(
		question_vector,
		dtype=np.float32
	).tobytes()

	#KNN Vector Search
	"""
		* means search every document, => perform vector operation,
		KNNtop-K return relevant closes vector,
		@embeddings means field field contain vector, our index has embedding VECTOR, compare with this field
		$vector use the embedding of the user's question

	"""
	#Build the KNN search
	search_command=("*=>"
		f"[KNN {how_many_results} @embeddings $vector AS distance_score]")

	#create query object
	query=Query(search_command)

	#Sort by similarity score
	query=query.sort_by("distance_score")

	#Return only these fields from Redis
	query=query.return_fields(
		"path",
		"file_type",
		"text",
		"distance_score"
	)

	#Enable Redisearch query dialect 2->it is used to enable vector search in KNN
	query=query.dialect(2)

	#Execute the search
	search_results=r.ft("idx:files").search(
		query,
		{
			"vector":query_vector_bytes
		}
	)
	#Only keep results that are close enough to be real matches
	good_results=[]
	for one_result in search_results.docs:
		distance=float(one_result.distance_score)
		if distance<=MAX_ALLOWED_DISTANCE:
			good_results.append(one_result)

	#Return matching documents
	return good_results

if __name__=="__main__":
	question=input("Ask: ")
	matching_chunks=search(question)
	print("\nTop Results\n")

	if not matching_chunks:
		print("No matching documents found.")

	for position, chunk in enumerate(matching_chunks, start=1):
		print("="*100)
		print(f"Result {i}")
		print("Path: ",chunk.path)
		print("Type: ",chunk.file_type)
		print("Score: ",chunk.score)
		print("Text: ")
		print(chunk.text)	
