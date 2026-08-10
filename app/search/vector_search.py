
import os
from app.storage.lancedb_store import get_table
from app.embeddings.embedding_router import generate_embedding

def search(question, how_many_results=5):
	table = get_table()
	if table is None:
		return []

	# Check if question mentions a specific file name or term in the database
	question_lower = question.lower()
	df = table.to_pandas()

	specific_chunks = []
	if not df.empty:
		for _, row in df.iterrows():
			path_str = str(row.get("path", ""))
			filename = os.path.basename(path_str).lower()
			name_without_ext = os.path.splitext(filename)[0]

			# If user asks specifically about a filename or file base name (e.g. "momo", "momo.mp4")
			if (filename in question_lower or (len(name_without_ext) > 2 and name_without_ext in question_lower)):
				specific_chunks.append(row.to_dict())

	# Semantic Vector Search
	question_vector = generate_embedding("text", question)
	try:
		semantic_results = table.search(question_vector).metric("cosine").limit(how_many_results).to_list()
	except Exception as e:
		print(f"Search Error: {e}")
		semantic_results = []

	# Combine specific file chunks with semantic search results (avoiding duplicates)
	seen_ids = set()
	combined_results = []

	for chunk in specific_chunks:
		chunk_id = chunk.get("chunk_id")
		if chunk_id and chunk_id not in seen_ids:
			seen_ids.add(chunk_id)
			combined_results.append(chunk)

	for chunk in semantic_results:
		chunk_id = chunk.get("chunk_id")
		if chunk_id and chunk_id not in seen_ids:
			seen_ids.add(chunk_id)
			combined_results.append(chunk)

	return combined_results[:how_many_results]
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
