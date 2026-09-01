import os
from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.storage.lancedb_store import get_hash_table, get_image_table
from app.query.query_analyzer import analyze_query
from app.llm.summary_client import summarize_text
from app.search.summary_search import search_for_summary, search_for_book_summary
from app.rag.image_summarizer import summarize_image
from app.search.object_search import search_images_by_object

def is_image_list_query(question):
	question_lower = question.lower()

	phrases = [
		"list all images",
		"list out all images",
		"list the images",
		"what images do you have",
		"which images do you have",
		"show all images",
		"show me all images",
		"all images",
		"images do you have"
	]

	return any(phrase in question_lower for phrase in phrases)


def is_object_search_query(question):
    question_lower = question.lower()

    phrases = [
        "find images containing",
        "find images with",
        "show images containing",
        "show images with",
        "images containing",
        "images with",
        "which images contain",
        "which images have",
        "find a picture of",
        "find pictures of",
        "find photos of",
        "show me images containing",
        "show me images with"
    ]

    return any(phrase in question_lower for phrase in phrases)


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

	if is_image_list_query(question):
		image_table = get_image_table()

		if image_table is None:
			return "No images are indexed.", [], 0

		image_df = image_table.to_pandas()

		if image_df.empty:
			return "No images are indexed.", [], 0

		image_paths = image_df["path"].tolist()

		image_names = sorted(
		    list(set(os.path.basename(path) for path in image_paths))
		)

		answer = "Images in your Personal Second Brain:\n"

		for i, name in enumerate(image_names, 1):
			answer += f"{i}. {name}\n"

		return answer, image_names, len(image_names)

	    # Step 1.6: Check for object-based image search
	if is_object_search_query(question):
		results = search_images_by_object(question)

		if not results:
			return (
                "No indexed image was found containing the requested object.",
                [],
                0
            )

		sources = sorted(
            list(
                set(
                    os.path.basename(result["path"])
                    for result in results
                )
            )
        )

		answer = "Images containing the requested object:\n"

		for i, source in enumerate(sources, 1):
			answer += f"{i}. {source}\n"

		return answer, sources, len(results)	

	# Step 2: Perform vector search
	analysis = analyze_query(question)
	# print("QUERY ANALYSIS:", analysis)

	if analysis["intent"] == "summarization":
		if analysis.get("chapter"):
			results = search_for_summary(question)

		else:
			results = search_for_book_summary(question)
	else:
	    results = search(question)

	# Retrieval decision determined by Python BEFORE calling Ollama
	if not results:
		return "Not found in the retrieved source.", [], 0
	# Handle image results using Qwen2.5-VL
	if results and results[0].get("file_type") == "image":
	    image_path = results[0]["path"]

	    try:
	        answer = summarize_image(image_path, question)
	    except Exception as e:
	        answer = f"Image VLM Error: {str(e)}"

	    sources = [os.path.basename(image_path)]

	    return answer, sources, len(results)

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

	if analysis["intent"] == "summarization":
	    answer = summarize_text(context_str)
	    return answer, sources, num_chunks

	# print("\n===== RETRIEVED CONTEXT =====")
	# for i, doc in enumerate(results, 1):
	#     print(f"\n--- Chunk {i} ---")
	#     print("Source:", os.path.basename(doc["path"]))
	#     print(doc["text"])
	# print("\n============================")

	# Summarization queries use the same retrieved chunks
	# but a dedicated summarization prompt.
	if analysis["intent"] == "summarization":
		answer = summarize(question, context_str)
		return answer, sources, num_chunks

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

def summarize(question, context):
    prompt = f"""You are a Personal Second Brain Assistant.

Summarize the retrieved content according to the user's request.

Rules:
- Use ONLY the retrieved context.
- Do not add information that is not present in the context.
- Keep the summary clear and concise.
- If the context does not contain enough information, say:
  "Not enough information found in the retrieved source."

Retrieved Context:
{context}

User Request:
{question}

Summary:"""

    try:
        return ask_llama(prompt)
    except Exception:
        try:
            return ask_gemini(prompt)
        except Exception as e:
            return f"LLM Error: {str(e)}"

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
		print("\n" + "=" * 75)

		question = input("\nAsk: ")
		if question.lower() == "exit":
			break
		if not question.strip():
			continue


		answer, sources, num_chunks = ask(question)
		print("\n┌" + "─" * 70 + "┐")

		print("│ QUESTION")
		print("│ " + question)

		print("│")

		print("│ ANSWER")
		for line in answer.split("\n"):
			print("│ " + line)

		print("│")

		print("│ SOURCE")

		if not sources or num_chunks == 0:
			print("│ No relevant source found.")

		else:
			for source in sources:
				print("│ " + source)

		print("│")

		print("│ RELEVANT CHUNKS")
		print("│ " + str(num_chunks))

		print("└" + "─" * 70 + "┘")
		# print("\n" + format_response(answer, sources, num_chunks))