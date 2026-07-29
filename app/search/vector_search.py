import numpy as np
from redis.commands.search.query import Query

from app.storage.redis_client import r
from app.embeddings.embedding_router import generate_embedding

def search(query_text, top_k=5):
	#Perform semantic search in Redis using vector similarity
	query_embedding=generate_embedding(
		"text",
		query_text
	)

	#Convert embedding to bytes
	query_vector=np.array(
		query_embedding,
		dtype=np.float32
	).tobytes()

	#KNN Vector Search
	query=(Query(f"*=>[KNN {top_k} @embeddings $vector AS score]"
		)
	.sort_by("score")
	.return_fields(
		"path",
		"file_type",
		"text",
		"score",
		)
		.dialect(2)
	)

	results=r.ft("idx:files").search(
		query,
		{
			"vector":query_vector
		}
	)
	return results.docs

if __name__=="__main__":
	query=input("Ask: ")
	results=search(query)
	print("\nTop Results\n")

	for i, doc in enumerate(results, start=1):
		print("="*100)
		print(f"Result {i}")
		print("Path: ",doc.path)
		print("Type: ",doc.file_type)
		print("Score: ",doc.score)
		print("Text: ")
		print(doc.text)	
