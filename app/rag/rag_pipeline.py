import os
import re
from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
from app.storage.lancedb_store import get_hash_table, get_image_table
from app.query.query_analyzer import analyze_query
from app.llm.summary_client import summarize_text
from app.search.summary_search import search_for_summary, search_for_book_summary
from app.rag.image_summarizer import summarize_image
from app.search.object_search import search_images_by_object
from app.search.face_search import is_face_search_query, search_images_by_face, extract_person_name_from_question

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
	question_lower = question.lower().strip()

	phrases = [
		# Existing patterns
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
		"show me images with",
		# Visual condition, multi-object & relationship patterns
		"image in which",
		"images in which",
		"picture in which",
		"pictures in which",
		"photo in which",
		"photos in which",
		"in which image",
		"in which images",
		"in which picture",
		"in which pictures",
		"in which photo",
		"in which photos",
		"in what image",
		"in what images",
		"image where",
		"images where",
		"picture where",
		"pictures where",
		"image showing",
		"images showing",
		"picture showing",
		"pictures showing",
		"list the name of image",
		"list the name of the image",
		"list the names of images",
		"list the names of the images",
		"which image has",
		"which images have",
		"which picture has",
		"which pictures have",
		"which image contains",
		"which images contain",
		"which picture contains",
		"which pictures contain",
		"which image shows",
		"which images show",
		"which picture shows",
		"which pictures show",
		"find images where",
		"find image where",
		"find pictures where",
		"find picture where",
		"find image containing",
		"find pictures containing",
		"find picture containing",
		"find image with",
		"find pictures with",
		"find picture with",
		"find a photo of",
		"show image containing",
		"show image with"
	]

	if any(phrase in question_lower for phrase in phrases):
		return True

	# Flexible regex pattern treating image/picture/photo and verbs/indicators as equivalent
	pattern_verb = r"\b(find|show|list|which|in\s+which|in\s+what)\b.*\b(image|images|picture|pictures|photo|photos)\b.*\b(contains?|have|has|shows?|where|in\s+which|that\s+has|that\s+have|that\s+contains?|showing|with|containing)\b"
	if re.search(pattern_verb, question_lower):
		return True

	pattern_noun_first = r"\b(image|images|picture|pictures|photo|photos)\b.*\b(contains?|have|has|shows?|where|in\s+which|that\s+has|that\s+have|that\s+contains?|showing|with|containing)\b"
	if re.search(pattern_noun_first, question_lower):
		return True

	return False


def ask(question, return_structured=False):
	"""
	Main RAG function: searches vector database and generates an answer.
	Returns: (answer_string, list_of_sources, num_relevant_chunks) by default,
	or structured dict when return_structured=True.
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
				ans = f"Yes, '{fn}' is indexed in your Personal Second Brain."
				if return_structured:
					return {"answer": ans, "sources": [fn], "num_chunks": 1, "type": "text", "images": []}
				return ans, [fn], 1

	# Step 1.4: Check for face-memory / person search
	if is_face_search_query(question):
		results = search_images_by_face(question)

		if not results:
			person_name = extract_person_name_from_question(question)
			if person_name:
				msg = f"No face memory record or matching image found for '{person_name.capitalize()}'. {person_name.capitalize()}'s face has not yet been registered in your Personal Second Brain."
			else:
				msg = "No indexed image was found matching the requested face / person."

			if return_structured:
				return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
			return msg, [], 0

		image_paths = sorted(
			list(
				set(
					r.get("image_path") or r.get("path")
					for r in results
					if (r.get("image_path") or r.get("path"))
				)
			)
		)

		sources = [os.path.basename(p) for p in image_paths]
		structured_images = [
			{
				"type": "image",
				"path": p,
				"source": os.path.basename(p)
			}
			for p in image_paths
		]

		answer_parts = ["Images matching the requested face / person:"]
		for p in image_paths:
			src = os.path.basename(p)
			answer_parts.append(f"\n![{src}](file://{p})\nSource: {src}\nPath: {p}")

		answer_str = "\n".join(answer_parts)

		if return_structured:
			return {
				"answer": answer_str,
				"sources": sources,
				"num_chunks": len(results),
				"type": "image",
				"images": structured_images
			}

		return answer_str, sources, len(results)

	if is_image_list_query(question):
		image_table = get_image_table()

		if image_table is None:
			msg = "No images are indexed."
			if return_structured:
				return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
			return msg, [], 0

		image_df = image_table.to_pandas()

		if image_df.empty:
			msg = "No images are indexed."
			if return_structured:
				return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
			return msg, [], 0

		image_paths = sorted(list(set(image_df["path"].dropna().tolist())))
		sources = [os.path.basename(p) for p in image_paths]
		structured_images = [
			{
				"type": "image",
				"path": p,
				"source": os.path.basename(p)
			}
			for p in image_paths
		]

		answer_parts = ["Images in your Personal Second Brain:"]
		for p in image_paths:
			src = os.path.basename(p)
			answer_parts.append(f"\n![{src}](file://{p})\nSource: {src}\nPath: {p}")

		answer_str = "\n".join(answer_parts)

		if return_structured:
			return {
				"answer": answer_str,
				"sources": sources,
				"num_chunks": len(sources),
				"type": "image",
				"images": structured_images
			}

		return answer_str, sources, len(sources)

	# Step 1.6: Check for object-based image search
	if is_object_search_query(question):
		results = search_images_by_object(question)

		if not results:
			msg = "No indexed image was found matching the requested condition."
			if return_structured:
				return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
			return msg, [], 0

		image_paths = sorted(
			list(
				set(
					result.get("path") or result.get("image_path")
					for result in results
					if (result.get("path") or result.get("image_path"))
				)
			)
		)

		sources = [os.path.basename(p) for p in image_paths]
		structured_images = [
			{
				"type": "image",
				"path": p,
				"source": os.path.basename(p)
			}
			for p in image_paths
		]

		answer_parts = ["Images matching the requested condition:"]
		for p in image_paths:
			src = os.path.basename(p)
			answer_parts.append(f"\n![{src}](file://{p})\nSource: {src}\nPath: {p}")

		answer_str = "\n".join(answer_parts)

		if return_structured:
			return {
				"answer": answer_str,
				"sources": sources,
				"num_chunks": len(results),
				"type": "image",
				"images": structured_images
			}

		return answer_str, sources, len(results)

	# Step 1.8: Check if query explicitly mentions an indexed image file
	image_table = get_image_table()
	if image_table is not None:
		image_df = image_table.to_pandas()
		if not image_df.empty:
			for idx, row in image_df.iterrows():
				img_path = row["path"]
				img_basename = os.path.basename(img_path)
				img_stem = os.path.splitext(img_basename)[0]
				if (len(img_stem) >= 3 and img_stem.lower() in question_lower) or img_basename.lower() in question_lower:
					try:
						vlm_summary = summarize_image(img_path, question)
					except Exception as e:
						vlm_summary = f"Image VLM Error: {str(e)}"
					src_name = img_basename
					answer_str = f"![{src_name}](file://{img_path})\n\nSource: {src_name}\nPath: {img_path}\n\n{vlm_summary}"
					if return_structured:
						return {
							"answer": answer_str,
							"sources": [src_name],
							"num_chunks": 1,
							"type": "image",
							"images": [{"type": "image", "path": img_path, "source": src_name}]
						}
					return answer_str, [src_name], 1

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
		msg = "Not found in the retrieved source."
		if return_structured:
			return {"answer": msg, "sources": [], "num_chunks": 0, "type": "text", "images": []}
		return msg, [], 0

	# Handle image results using Qwen2.5-VL
	if results and results[0].get("file_type") == "image":
		image_path = results[0]["path"]

		try:
			vlm_summary = summarize_image(image_path, question)
		except Exception as e:
			vlm_summary = f"Image VLM Error: {str(e)}"

		sources = [os.path.basename(image_path)]
		src_name = sources[0]
		answer_str = f"![{src_name}](file://{image_path})\n\nSource: {src_name}\nPath: {image_path}\n\n{vlm_summary}"

		if return_structured:
			return {
				"answer": answer_str,
				"sources": sources,
				"num_chunks": len(results),
				"type": "image",
				"images": [{"type": "image", "path": image_path, "source": src_name}]
			}

		return answer_str, sources, len(results)

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

	if return_structured:
		extracted_chunks = []
		for item in results[:3]:
			if isinstance(item, (tuple, list)) and len(item) > 0:
				doc = item[0]
			else:
				doc = item
			if isinstance(doc, dict):
				extracted_chunks.append(doc.get("text", ""))
			else:
				extracted_chunks.append(str(doc))

		return {
			"answer": answer,
			"sources": sources,
			"num_chunks": num_chunks,
			"type": "text",
			"chunks": extracted_chunks,
			"images": []
		}

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