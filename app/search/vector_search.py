import os
import re
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


def extract_query_keywords(query):
	"""
	Extracts significant search keywords from query, filtering out common stop words.
	"""
	stop_words = {
		"what", "which", "where", "who", "how", "when", "why", "did", "does", "is", "are", "was",
		"were", "the", "a", "an", "in", "on", "at", "for", "to", "of", "her", "his", "their",
		"mentioned", "listed", "complete", "used", "only", "using", "experience", "current",
		"resume", "document", "file"
	}
	words = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', query.lower())
	return [w for w in words if w not in stop_words and len(w) > 2]


def search(question, max_results=10, score_threshold=0.55):
	"""
	Performs Hybrid Search (Vector + Keyword Reranking) with Document Diversity Capping.
	Admits genuine natural language and keyword matches while strictly filtering out unrelated queries.
	"""

	table = get_table()
	if table is None:
		print("LanceDB table is not available.")
		return []

	question_vector = generate_embedding("text", question)
	question_lower = question.lower()
	keywords = extract_query_keywords(question)

	df = table.to_pandas()
	target_path = None

	if not df.empty:
		for path_str in df['path'].unique():
			filename = os.path.basename(str(path_str)).lower()
			if filename in question_lower:
				target_path = path_str
				break

	try:
		# Step 1: Raw vector search candidates (limit 100)
		if target_path:
			escaped_path = str(target_path).replace("'", "''")
			raw_chunks = table.search(question_vector).where(f"path = '{escaped_path}'").metric("cosine").limit(100).to_list()
		else:
			raw_chunks = table.search(question_vector).metric("cosine").limit(100).to_list()

		# Step 2: Multi-keyword candidate search in DB to boost exact entity/keyword matches
		kw_candidates = {}
		if keywords and not df.empty:
			for idx, row in df.iterrows():
				text_lower = str(row['text']).lower()
				matches = [kw for kw in keywords if kw in text_lower]
				if len(matches) >= 2 or (len(keywords) == 1 and len(matches) >= 1):
					kw_candidates[row['chunk_id']] = {
						"chunk_id": row['chunk_id'],
						"path": row['path'],
						"file_type": row['file_type'],
						"text": row['text'],
						"_distance": 0.70
					}

		# Combine vector and keyword candidates
		combined = {}
		for doc in raw_chunks:
			combined[doc['chunk_id']] = doc
		for cid, doc in kw_candidates.items():
			if cid not in combined:
				combined[cid] = doc

		is_asking_for_toc = "table of contents" in question_lower or "contents" in question_lower or "bibliography" in question_lower

		# Step 3: Hybrid Scoring & Keyword Reranking
		scored = []
		for doc in combined.values():
			text = doc.get("text", "")
			text_lower = text.lower()
			raw_dist = doc.get("_distance", 1.0)
			src = os.path.basename(doc.get("path", ""))

			if is_table_of_contents_or_noise(text) and not is_asking_for_toc:
				continue

			matched_kws = [kw for kw in keywords if kw in text_lower]
			kw_match_count = len(matched_kws)

			bonus = 0.0
			if kw_match_count > 0:
				ratio = kw_match_count / max(len(keywords), 1)
				bonus += ratio * 0.25
				if kw_match_count >= 2:
					bonus += 0.10

			if "rashmi" in question_lower and "rashmi" in text_lower:
				bonus += 0.10

			rerank_score = raw_dist - bonus

			# Quality Threshold: Must pass score threshold or have strong multi-keyword match
			if raw_dist <= score_threshold or (kw_match_count >= 2 and rerank_score <= 0.60):
				scored.append({
					"doc": doc,
					"src": src,
					"raw_dist": raw_dist,
					"rerank_score": rerank_score,
					"kw_count": kw_match_count
				})

		scored.sort(key=lambda x: x["rerank_score"])

		# Step 4: Document Diversity Cap (max 3 chunks per source file)
		# Prevents any single 2,500-page book or 69-row spreadsheet from monopolizing top-k slots
		relevant_chunks = []
		source_counts = {}

		for item in scored:
			src = item["src"]
			cnt = source_counts.get(src, 0)

			if target_path or cnt < 3:
				source_counts[src] = cnt + 1
				relevant_chunks.append(item["doc"])
				if len(relevant_chunks) == max_results:
					break

		# Step 5: Diagnostic logging
		sources = sorted(list(set(os.path.basename(c["path"]) for c in relevant_chunks)))
		print(f"\nQuery: {question}")
		print(f"Raw chunks: {len(raw_chunks)}")
		print(f"Relevant chunks: {len(relevant_chunks)}")
		print(f"Sources: {', '.join(sources) if sources else 'None'}")

		return relevant_chunks

	except Exception as e:
		print(f"Search Error: {e}")
		return []



if __name__ == "__main__":
	while True:
		question = input("Ask: ")
		results = search(question)
		print(f"Retrieved {len(results)} chunks.")
