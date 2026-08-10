from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
import os

from app.storage.lancedb_store import get_hash_table, get_table

def ask(question):
	# 1. Check for metadata / file existence / file list questions
	question_lower = question.lower()
	is_existence_query = any(pattern in question_lower for pattern in [
		"do you have", "do i have", "which video", "which file", "which document",
		"what files", "list files", "available video", "available file", "show me files"
	])

	if is_existence_query:
		hash_table = get_hash_table()
		table = get_table()

		all_paths = set()
		if hash_table is not None:
			hdf = hash_table.to_pandas()
			if not hdf.empty:
				all_paths.update(hdf["path"].tolist())
		if table is not None:
			df = table.to_pandas()
			if not df.empty:
				all_paths.update(df["path"].tolist())

		file_names = sorted(list(set(os.path.basename(p) for p in all_paths)))

		# Check if asking about specific file existence (e.g. "momo")
		for fn in file_names:
			base_fn = os.path.splitext(fn)[0].lower()
			if base_fn in question_lower or fn.lower() in question_lower:
				return f"Yes, '{fn}' is indexed in your Personal Second Brain.", [fn]

		if "video" in question_lower:
			videos = [fn for fn in file_names if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
			if videos:
				return f"Yes, the following video(s) are available: {', '.join(videos)}", videos
			else:
				return "No video files are currently indexed.", []

		if "file" in question_lower or "list" in question_lower:
			if file_names:
				return f"The following files are indexed in your Personal Second Brain: {', '.join(file_names)}", file_names

	# 2. Search similar chunks
	results = search(question)

	if not results:
		return "I could not find any relevant information in the indexed documents.", []

	# Combine all retrieved text
	context = []
	retrieved_sources = set()

	for doc in results:
		fname = os.path.basename(doc['path'])
		retrieved_sources.add(fname)
		context.append(f"Source File: {fname}\n{doc['text']}")

	context_str = "\n\n".join(context)

	# Make prompt
	prompt = f"""
		You are a Personal Second Brain Assistant.

		Answer the user's question using the retrieved information below.

		Guidelines:
			- Use the retrieved document content and the document name ("Source File") to answer the question.
			- If the user asks about the content or summary of a file, describe or summarize whatever text/transcript is provided in the Context (including transcribed audio/speech).
			- If the context contains no relevant information at all to answer the question, reply:
			"I could not find that information in the indexed documents."

		Retrieved Information:	

		Context: {context_str}

		Question: {question}

		Answer: """

	try:
		answer = ask_llama(prompt)
	except Exception as e:
		try:
			answer = ask_gemini(prompt)
		except Exception as e2:
			return f"LLM Error: {str(e2)}", []

	# Only report sources if answer is not a failure response
	if "I could not find" in answer:
		sources = []
	else:
		sources = sorted(list(retrieved_sources))

	return answer, sources	

if __name__=="__main__":
	while True:
		question=input("\n\n\nAsk: ")
		if question.lower()=="exit":
			print("\nThank you for using Personal Second Brain!")
			break

		answer, sources=ask(question)
		
		print("\n"+"="*100)
		print("🧠 Personal Second Brain")
		print("="*100)

		print("\nQuery: ")
		print("-"*100)
		print(question)

		print("\nAnswer:")
		print("-"*100)
		print(answer)	
		print("-"*100)

		# print("\nSource File:")
		# for source in sources:
		# 	print(f"-{source}")	