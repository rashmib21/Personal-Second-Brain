
from app.storage.lancedb_store import table
from app.embeddings.embedding_router import generate_embedding
	


def search(question, how_many_results=5):
	#convert user query into embeddings
	question_vector=generate_embedding(
		"text",
		question
	)

	#Semantic Search
	results=(table.search(question_vector).metric("cosine").limit(how_many_results).to_list())
	
	if not results:
		return []


	#Return matching documents
	return results

if __name__=="__main__":
	
	while True: 
		question=input("Ask: ")
		matching_chunks=search(question)
		print("\nTop Results\n")

		if not matching_chunks:
			print("No matching documents found.")

		for position, chunk in enumerate(matching_chunks, start=1):
			print("="*100)
			print(f"Result {position}")
			print("Path: ",chunk["path"])
			print("Type: ",chunk["file_type"])
			print("Score: ",chunk["_distance"])

			print("\nText: ")
			print("="*100)
			print(chunk["text"])
			print()	
