import os
from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.storage.lancedb_store import get_hash_table


def ask(question):
	"""
	Main RAG function: searches vector database and generates an answer using Ollama.
	Returns: (answer_string, list_of_sources, num_relevant_chunks)
	"""
	question_lower = question.lower()

	# Step 1: Check for file list / file existence questions
	is_file_list_query = any(phrase in question_lower for phrase in [
		"do you have", "do i have", "which file", "which document",
		"what files", "list files", "show me files"
	])

	if is_file_list_query:
		hash_table = get_hash_table()
		all_paths = []
		if hash_table is not None:
			hdf = hash_table.to_pandas()
			if not hdf.empty:
				all_paths = hdf["path"].tolist()

		file_names = sorted(list(set(os.path.basename(p) for p in all_paths)))

		for fn in file_names:
			if fn.lower() in question_lower:
				return f"Yes, '{fn}' is indexed in your Personal Second Brain.", [fn], 1

		
	# Step 2: Perform vector search
	results = search(question)
	

	# Retrieval decision determined by Python BEFORE calling Ollama
	if not results:
		return "Not found in the retrieved source.", [], 0
	num_chunks = len(results)

	# Step 3: Build context string and collect source filenames from python results
	context_list = []
	retrieved_sources = set()

	for doc in results:
		filename = os.path.basename(doc["path"])
		retrieved_sources.add(filename)
		context_list.append(doc["text"])

	sources = sorted(list(retrieved_sources))
	context_str = "\n\n".join(context_list)

	# Step 4: Build prompt for LLM
	prompt = f"""You are a Personal Second Brain Assistant.

Answer the user's question using ONLY the retrieved context below.

Guidelines:
- Rely strictly on the retrieved document content in the Context.
- You may synthesize an answer across multiple retrieved chunks.
- Do not use general outside knowledge or make up information.
- If the retrieved context genuinely does not contain enough information to answer the question, reply exactly:
  "Not found in the retrieved source."

Retrieved Context:
{context_str}

Question: {question}

Answer:"""

	# Step 5: Ask LLM (Ollama first, Gemini fallback)
	try:
		answer = ask_llama(prompt)
	except Exception:
		try:
			answer = ask_gemini(prompt)
		except Exception as e:
			answer = f"LLM Error: {str(e)}"

	return answer, sources, num_chunks


def format_response(answer, sources, num_chunks=0):
	"""
	Formats the final answer, source attribution, and relevant chunk count.
	"""
	output = f"Answer:\n{answer}\n\n"

	if not sources or num_chunks == 0:
		output += "Source:\nNo relevant source found.\n\n"
	elif len(sources) == 1:
		output += f"Source:\n{sources[0]}\n\n"
	else:
		output += "Sources:\n" + "\n".join(f"- {s}" for s in sources) + "\n\n"

	output += f"Relevant chunks:\n{num_chunks}"

	return output


if __name__ == "__main__":
	while True:
		question = input("\nAsk: ")
		if question.lower() == "exit":
			break

		answer, sources, num_chunks = ask(question)
		print("\n" + format_response(answer, sources, num_chunks))