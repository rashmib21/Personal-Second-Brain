
from app.storage.lancedb_store import table
from app.embeddings.embedding_router import generate_embedding
	
MAX_ALLOWED_DISTANCE = 0.5

def search(question, how_many_results=5):
	#convert user query into embeddings
	question_vector=generate_embedding(
		"text",
		question
	)

	results=(table.search(question_vector).metric("cosine").limit(how_many_results).to_list())
	
	#Only keep results that are close enough to be real matches
	good_results=[]
	for one_result in results:

		distance=float(one_result["_distance"])
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
		print(f"Result {position}")
		print("Path: ",chunk["path"])
		print("Type: ",chunk["file_type"])
		print("Score: ",chunk["_distance"])
		print("Text: ")
		print(chunk["text"])	
