import os
from app.storage.lancedb_store import get_table
from app.embeddings.embedding_router import generate_embedding


def is_table_of_contents_or_noise(text):
	"""
	Checks if a chunk contains Table of Contents dotted lines (. . . 419)
	or publisher blurb reviews that crowd out actual body text.
	"""
	text_lower = text.lower()

	# Chunks with 15 or more periods are usually Table of Contents or Index page numbers
	if text.count(".") >= 15:
		return True

	# Publisher review headers or Table of Contents section headings
	if "contents in detail" in text_lower or "reviews for how linux works" in text_lower or "420 bibliography" in text_lower:
		return True

	return False


def search(question, max_results=10, score_threshold=0.50):
	"""
	Performs semantic vector search and returns up to max_results relevant chunks.
	"""
	table = get_table()
	if table is None:
		print("LanceDB table is not available.")
		return []

	question_vector = generate_embedding("text", question)
	question_lower = question.lower()

	# Step 1: Check if question mentions a specific file name (e.g. linux.pdf)
	df = table.to_pandas()
	target_path = None

	if not df.empty:
		for path_str in df['path'].unique():
			filename = os.path.basename(str(path_str)).lower()
			if filename in question_lower:
				target_path = path_str
				break

	# Step 2: Fetch candidates from LanceDB
	try:
		if target_path:
			escaped_path = str(target_path).replace("'", "''")
			raw_chunks = table.search(question_vector).where(f"path = '{escaped_path}'").metric("cosine").limit(50).to_list()
		else:
			raw_chunks = table.search(question_vector).metric("cosine").limit(50).to_list()

		# Step 3: Check if question is specifically asking for Table of Contents or Bibliography
		is_asking_for_toc = "table of contents" in question_lower or "contents" in question_lower or "bibliography" in question_lower

		# Step 4: Prioritize main body chunks over Table of Contents noise
		body_chunks = []
		noise_chunks = []

		for doc in raw_chunks:
			text = doc.get("text", "")
			if is_table_of_contents_or_noise(text) and not is_asking_for_toc:
				noise_chunks.append(doc)
			else:
				body_chunks.append(doc)

		# Combine: body chunks first, noise chunks after
		ordered_chunks = body_chunks + noise_chunks

		# Step 5: Filter by relevance score threshold (distance <= 0.50)
		relevant_chunks = []
		for doc in ordered_chunks:
			score = doc.get("_distance", 1.0)
			if score <= score_threshold:
				relevant_chunks.append(doc)

			# Stop when we reach max_results limit (e.g. 10)
			if len(relevant_chunks) == max_results:
				break

		# Step 6: Diagnostic logging
		sources = sorted(list(set(os.path.basename(c["path"]) for c in relevant_chunks)))
		print(f"\nQuery: {question}")
		print(f"Raw chunks: {len(raw_chunks)}")
		print(f"Relevant chunks: {len(relevant_chunks)}")
		print(f"Sources: {', '.join(sources) if sources else 'None'}")
		print(f"Sending {len(relevant_chunks)} chunks to Ollama.")

		return relevant_chunks

	except Exception as e:
		print(f"Search Error: {e}")
		return []


if __name__ == "__main__":
	while True:
		question = input("Ask: ")
		results = search(question)
		print(f"Retrieved {len(results)} chunks.")
