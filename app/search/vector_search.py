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
