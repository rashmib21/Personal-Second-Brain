import numpy as np
from redis.commands.search.query import Query

from app.storage.redis_store import r
from app.embeddings.embedding_router import generate_embedding

def search(query_text, top_k=5):
	#convert user query into embeddings
	query_embedding=generate_embedding(
		"text",
		query_text
	)

	#Convert embedding list to bytes
	query_vector=np.array(
		query_embedding,
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
	knn_query=("*=>"
		f"[KNN {top_k} @embeddings $vector AS score]")

	#create query object
	redis_query=Query(knn_query)

	#Sort by similarity score
	redis_query=redis_query.sort_by("score")

	#Return only these fields from Redis
	redis_query=redis_query.return_fields(
		"path",
		"file_type",
		"text",
		"score"
	)

	#Enable Redisearch query dialect 2->it is used to enable vector search in KNN
	redis_query=redis_query.dialect(2)

	#Execute the search
	results=r.ft("idx:files").search(
		redis_query,
		{
			"vector":query_vector
		}
	)

	#Return matching documents
	return results.docs

if __name__=="__main__":
	query=input("Ask: ")
	results=search(query)
	print("\nTop Results\n")

	if not results:
		print("No matching documents found.")

	for i, doc in enumerate(results, start=1):
		print("="*100)
		print(f"Result {i}")
		print("Path: ",doc.path)
		print("Type: ",doc.file_type)
		print("Score: ",doc.score)
		print("Text: ")
		print(doc.text)	
