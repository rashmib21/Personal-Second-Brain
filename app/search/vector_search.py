import os
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from app.storage.lancedb_store import get_table, get_image_table
from app.embeddings.embedding_router import generate_embedding
from app.embeddings.image_embedding import embed_image, get_clip_model
from config import DEBUG


#takes a question and one chunk of text together as a pair, and directly outputs how relevant that chunk actually is to the question — instead of comparing separate embeddings or matching separate words, it reads both at once and judges the match itself.
from sentence_transformers import CrossEncoder 
from nltk.stem import PorterStemmer

reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
stemmer = PorterStemmer()


def simple_stem(word):
	return stemmer.stem(word)


def is_noise_chunk(text):
	#Skip table of contents/index pages that just clutter results
	text_lower=text.lower()

	#Dotted line like chpater 3.......42(table of content)
	if text.count(".")>=15:
		return True

	#lots of short lines ending in numbers like p no. which is typical of a table of contents or an index page
	lines=text.split("\n")
	lines_ending_in_number=0
	for line in lines:
		line=line.strip()
		if line and line[-1].isdigit():
			lines_ending_in_number+=1

	if len(lines)>5 and lines_ending_in_number/len(lines)>=0.5:
		return True

	#generic heading words that indicate a TOC/index section, not tied to any specific book's title or wording 
	generic_noise_headings=["table of contents","bibliography", "index of terms"]
	for heading in generic_noise_headings:
		if heading in text_lower:
			return True
	return False					


def get_words(text):
	#break text into simple lowercase words. 
	words=re.findall(r'\b[a-zA-Z0-9_\-\.]+\b',text.lower())
	words=[w for w in words if w not in ENGLISH_STOP_WORDS]
	words=[simple_stem(w) for w in words]
	return words

def rerank(question, candidate_chunks):
	#candidate_chunks=list of chunks dicts that receive after RRf
	pairs=[]
	for chunk in candidate_chunks:
		pair=(question, chunk['text'])
		pairs.append(pair)

	#ask the cross encoder model to score every pair
	#higher score=more relevant to the question
	scores=reranker.predict(pairs)

	#attach each score to its matching chunk
	scored_chunks=[]
	for i in range(len(candidate_chunks)):
		chunk=candidate_chunks[i]
		score=scores[i]
		scored_chunks.append((chunk,score))

	#sort manually so the highest score comes first
	def get_score(pair):
		#pair looks like (chunks, score), we want to sort by score, which is pair[1]
		return pair[1]
	scored_chunks.sort(key=get_score, reverse=True)
	
	return scored_chunks		

def is_image_question(question):
	question_lower = question.lower()
	image_words = [
		"image",
		"picture",
		"photo",
		"visual",
		"diagram",
		"chart",
		"figure",

		"in the image",
		"in this image",
		"from the image",
		"from this image",
		"shown in the image",
		"shown in this image",
		"shown in the picture",
		"shown in this picture",
		"according to the image",
		"according to this image",
		"mentioned in the image",
		"mentioned in this image",

		"what is happening",
		"what does the image show",
		"what information is shown",
		"describe the image",
	    "describe this image",

		"which job",
		"which jobs",
		"what job",
		"what jobs",
	]

	return any(word in question_lower for word in image_words)


def search(question, max_results=10, analysis=None, preferred_sources=None, rejected_sources=None):
	"""
	Source-aware & modality-aware hybrid retrieval with RRF and CrossEncoder reranking.
	Enforces explicit source filter BEFORE semantic search / reranking.
	Unifies candidate lookup across documents and image_documents tables.
	"""
	from app.query.query_analyzer import analyze_query

	# Step 1: Analyze query if analysis object is not passed
	if analysis is None:
		analysis = analyze_query(question)

	query_modality = analysis.get("modality", "all")
	source_hint = analysis.get("source_hint")

	preferred_set = set()
	if preferred_sources:
		for src in preferred_sources:
			preferred_set.add(src.lower())

	if source_hint and not str(source_hint).startswith("UNRESOLVED_"):
		preferred_set.add(source_hint.lower())

	rejected_set = set()
	if rejected_sources:
		for src in rejected_sources:
			rejected_set.add(src.lower())

	# Step 2: Fetch all indexed chunks from LanceDB (documents and image_documents tables)
	doc_table = get_table()
	image_table = get_image_table()

	all_indexed_rows = []

	if doc_table is not None and doc_table.count_rows() > 0:
		df_doc = doc_table.to_pandas()
		for i in range(len(df_doc)):
			r = df_doc.iloc[i]
			all_indexed_rows.append({
				"chunk_id": str(r.get("chunk_id", f"doc_{i}")),
				"path": str(r.get("path", "")),
				"file_type": str(r.get("file_type", "document")),
				"text": str(r.get("text", ""))
			})

	if image_table is not None and image_table.count_rows() > 0:
		df_img = image_table.to_pandas()
		for i in range(len(df_img)):
			r = df_img.iloc[i]
			all_indexed_rows.append({
				"chunk_id": str(r.get("chunk_id", f"img_{i}")),
				"path": str(r.get("path", "")),
				"file_type": "image",
				"text": str(r.get("text", ""))
			})

	if len(all_indexed_rows) == 0:
		if DEBUG:
			print("No indexed data found in database.")
		return []

	# Filter out noise chunks and rejected sources
	clean_rows = []
	for row in all_indexed_rows:
		r_base = os.path.basename(row["path"]).lower()
		if rejected_set and r_base in rejected_set and r_base not in preferred_set:
			continue
		if not is_noise_chunk(row["text"]):
			clean_rows.append(row)

	# Step 3: Enforce Hard Pre-Retrieval Source Filter BEFORE Search/Reranking
	canonical_source_id = analysis.get("canonical_source_id")
	resolved_source = source_hint
	source_exists = False

	unresolved_flag = analysis.get("unresolved_explicit_source", False)
	if (source_hint and str(source_hint).startswith("UNRESOLVED_")) or unresolved_flag:
		if DEBUG:
			print(f"HARD SOURCE ROUTING: Source '{source_hint}' is unresolved. Returning 0 candidate chunks to prevent cross-source fallback.")
		return []

	candidate_rows = []

	if source_hint or canonical_source_id:
		# For source-specific requests, restrict candidates strictly to the resolved source
		target_names = set()
		if source_hint:
			target_names.add(source_hint.lower())
			target_names.add(os.path.basename(source_hint).lower())
		if canonical_source_id:
			target_names.add(canonical_source_id.lower())
			target_names.add(os.path.basename(canonical_source_id).lower())

		for row in clean_rows:
			r_path_lower = row["path"].lower()
			r_base = os.path.basename(row["path"]).lower()
			if r_base in target_names or r_path_lower in target_names:
				candidate_rows.append(row)

		if len(candidate_rows) > 0:
			source_exists = True
			resolved_source = os.path.basename(candidate_rows[0]["path"])
		else:
			source_exists = False
			if DEBUG:
				print(f"HARD SOURCE ROUTING: Source '{source_hint}' exists but yielded 0 usable chunks. Zero fallback.")
			return []
	elif preferred_set:
		# For non-source-specific queries with preferences, use preferred_set
		target_names = set(preferred_set)
		for row in clean_rows:
			r_path_lower = row["path"].lower()
			r_base = os.path.basename(row["path"]).lower()
			if r_base in target_names or r_path_lower in target_names:
				candidate_rows.append(row)

		if len(candidate_rows) > 0:
			source_exists = True
			resolved_source = os.path.basename(candidate_rows[0]["path"])
		else:
			source_exists = False
			return []

		# Hard Runtime Validation: Ensure EVERY candidate matches target source
		for cand in candidate_rows:
			c_base = os.path.basename(cand["path"]).lower()
			c_path = cand["path"].lower()
			if c_base not in target_names and c_path not in target_names:
				raise RuntimeError(
					f"HARD SOURCE ROUTING VIOLATION: Candidate '{cand['path']}' does not match target source '{target_names}'"
				)

	elif query_modality != "all":
		# Explicit modality routing.
		# Only currently supported application modalities enter retrieval.
		if query_modality == "image":
			img_exts = [".jpg", ".jpeg", ".png", ".webp"]

			for row in clean_rows:
				p_low = row["path"].lower()
				ftype = row["file_type"].lower()

				if (
					ftype == "image"
					or any(p_low.endswith(ext) for ext in img_exts)
				):
					candidate_rows.append(row)

		elif query_modality in {
			"pdf",
			"document",
			"docx",
			"spreadsheet",
			"text",
			"presentation",
		}:
			supported_document_exts = {
				".pdf",
				".doc",
				".docx",
				".xls",
				".xlsx",
				".csv",
				".ods",
				".txt",
				".ppt",
				".pptx",
			}

			supported_document_types = {
				"pdf",
				"document",
				"docx",
				"spreadsheet",
				"text",
				"presentation",
			}

			for row in clean_rows:
				p_low = row["path"].lower()
				ftype = row["file_type"].lower()

				if (
					ftype in supported_document_types
					or any(p_low.endswith(ext) for ext in supported_document_exts)
				):
					candidate_rows.append(row)

		else:
			# Unknown or unsupported modality.
			# Never silently search the entire database.
			candidate_rows = []

	else:
		# "all" means all CURRENTLY SUPPORTED searchable content.
		# Historical audio/video records in LanceDB are excluded.
		supported_exts = {
			".pdf",
			".doc",
			".docx",
			".xls",
			".xlsx",
			".csv",
			".ods",
			".txt",
			".ppt",
			".pptx",
			".jpg",
			".jpeg",
			".png",
			".webp",
		}

		supported_types = {
			"pdf",
			"document",
			"docx",
			"spreadsheet",
			"text",
			"presentation",
			"image",
		}

		for row in clean_rows:
			p_low = row["path"].lower()
			ftype = row["file_type"].lower()

			if (
				ftype in supported_types
				or any(p_low.endswith(ext) for ext in supported_exts)
			):
				candidate_rows.append(row)

	if len(candidate_rows) == 0:
		return []


	# Step 4: Dense Vector Embedding Search
	# Search only candidates from the compatible text embedding table.
	# Image CLIP embeddings must NOT be compared with text embeddings.

	question_vector = generate_embedding("text", question)

	dense_rank = {}

	if candidate_rows and query_modality != "image" and doc_table is not None:
	    candidate_ids = {
	        str(row["chunk_id"])
	        for row in candidate_rows
	    }

	    dense_results = (
	        doc_table
	        .search(question_vector)
	        .metric("cosine")
	        .limit(min(300, doc_table.count_rows()))
	        .to_list()
	    )

	    rank_number = 1

	    for doc in dense_results:
	        doc_id = str(doc.get("chunk_id", ""))

	        if doc_id in candidate_ids:
	            dense_rank[doc_id] = rank_number
	            rank_number += 1

	    # Candidates missed by dense retrieval get a deterministic fallback rank.
	    for row in candidate_rows:
	        chunk_id = str(row["chunk_id"])

	        if chunk_id not in dense_rank:
	            dense_rank[chunk_id] = len(dense_rank) + 1

	# Step 5: Lexical (BM25) search ranking across candidate_rows
	all_words_list = []
	for row in candidate_rows:
		all_words_list.append(get_words(row["text"]))

	bm25 = BM25Okapi(all_words_list)
	question_words = get_words(question)
	bm25_scores = bm25.get_scores(question_words)

	bm25_pairs = []
	for i in range(len(candidate_rows)):
		chunk_id = candidate_rows[i]["chunk_id"]
		score = bm25_scores[i]
		bm25_pairs.append((chunk_id, score))

	def get_second_item(pair):
		return pair[1]

	bm25_pairs.sort(key=get_second_item, reverse=True)

	lexical_rank = {}
	rank_number = 1
	for chunk_id, score in bm25_pairs:
		lexical_rank[chunk_id] = rank_number
		rank_number += 1

	
	for r in candidate_rows:
		if r["chunk_id"] not in dense_rank:
			dense_rank[r["chunk_id"]] = len(dense_rank) + 1

	chunk_data_by_id = {r["chunk_id"]: r for r in candidate_rows}

	# Step 6: Reciprocal Rank Fusion (RRF)
	k = 60
	all_candidate_chunk_ids = list(chunk_data_by_id.keys())

	combined_score = []
	for chunk_id in all_candidate_chunk_ids:
		score = 0.0
		if chunk_id in dense_rank:
			score += 1.0 / (k + dense_rank[chunk_id])

		if chunk_id in lexical_rank:
			score += 1.0 / (k + lexical_rank[chunk_id])

		combined_score.append((chunk_id, score))

	combined_score.sort(key=get_second_item, reverse=True)

	rrf_candidates = []
	for chunk_id, score in combined_score[:30]:
		if chunk_id in chunk_data_by_id:
			rrf_candidates.append(chunk_data_by_id[chunk_id])

	# Step 7: CrossEncoder Reranking
	reranked = rerank(question, rrf_candidates)

	final_chunks = []
	seen_texts = set()

	if reranked:
		for doc, score in reranked:
			if score < -8.0 and not preferred_set:
				continue
			normalized_text = re.sub(r"\s+", " ", doc["text"].strip().lower())
			if normalized_text in seen_texts:
				continue
			seen_texts.add(normalized_text)
			final_chunks.append(doc)
			if len(final_chunks) >= max_results:
				break

	if not final_chunks and candidate_rows:
		final_chunks = candidate_rows[:max_results]

	return final_chunks
					

#------Image Search-----------
def search_images(image_path, max_results=10):
	#Search images using CLIP image embeddings.
	#image_path: path of the query image.
	#returns: most visually similar images from image_documents.

	image_table=get_image_table()

	if image_table is None:
		if DEBUG:
			print("Image table is not available")
		return []

	#Generate 512-D CLIP embedding for query image
	query_embedding=embed_image(image_path)

	#Search only inside image_documents
	results=(
			image_table
			.search(query_embedding)
			.metric("cosine")
			.limit(max_results)
			.to_list()
	)

	if DEBUG:
		print("\nImage Query: ",image_path)
		print("Relevant images: ", len(results))

		for result in results:
			print("Source: ",os.path.basename(result['path']))
	return results	

#-------Image Search by text--------
def search_images_by_text(question, max_results=5):

	#Search image documents using a text query
	image_table=get_image_table()

	if image_table is None:
		if DEBUG:
			print("Image table is not available.")
		return []

	#Generate 512-D CLIP text embedding
	clip_model=get_clip_model()
	query_embedding=clip_model.encode(question).tolist()

	results=(
		image_table
		.search(query_embedding)
		.metric("cosine")
		.limit(max_results)
		.to_list()
	)
	filtered_results = results
	# is_image_query = any(w in question.lower() for w in ["image", "picture", "photo", "diagram", "chart", "figure"])
	# filtered_results = []
	# for res in results:
	# 	dist = res.get("_distance", 1.0)
	# 	if (is_image_query and dist < 0.85) or (not is_image_query and dist < 0.55):
	# 		filtered_results.append(res)

	if DEBUG:
		print("\nImage Text Query: ", question)
		print("Relevant Images: ", len(filtered_results))

		for result in results:
			print("Source: ", os.path.basename(result['path']))
	return results		


def get_full_content_for_source(source_path):
    """
    Retrieve all stored chunks for one source in deterministic sequence order.

    The function:
    - matches only the requested source
    - detects explicit chunk-number gaps
    - detects duplicate chunk IDs
    - preserves chunk content
    - removes only structural boundary overlap
    - reports completeness honestly
    """
    doc_table = get_table()

    empty_metrics = {
        "expected_chunks": 0,
        "fetched_chunks": 0,
        "missing_chunks": 0,
        "duplicate_chunks": 0,
        "reconstructed_chunks": 0,
        "is_complete": False,
        "ordering_guarantee": "none",
        "missing_chunk_indices": []
    }

    if not source_path or doc_table is None or doc_table.count_rows() == 0:
        return "", 0, empty_metrics

    df_doc = doc_table.to_pandas()

    if df_doc.empty:
        return "", 0, empty_metrics

    target_base = os.path.basename(source_path).lower()
    source_path_lower = str(source_path).lower()

    matching_rows = []

    for i in range(len(df_doc)):
        row = df_doc.iloc[i]

        r_path = str(row.get("path", ""))
        r_path_lower = r_path.lower()
        r_base = os.path.basename(r_path).lower()

        if (
            r_base == target_base
            or r_path_lower == source_path_lower
        ):
            matching_rows.append(row.to_dict())

    if not matching_rows:
        return "", 0, empty_metrics

    indexed_rows = []
    unindexed_rows = []

    for row in matching_rows:
        chunk_id = str(row.get("chunk_id", ""))

        match = re.search(r"_chunk_(\d+)", chunk_id)

        if match:
            indexed_rows.append((int(match.group(1)), row))
        else:
            unindexed_rows.append(row)

    duplicate_chunks = 0

    if indexed_rows:
        indexed_rows.sort(key=lambda item: item[0])

        observed_indices = [index for index, _ in indexed_rows]

        duplicate_indices = {
            index
            for index in observed_indices
            if observed_indices.count(index) > 1
        }

        duplicate_chunks = sum(
            observed_indices.count(index) - 1
            for index in duplicate_indices
        )

        min_index = min(observed_indices)
        max_index = max(observed_indices)

        expected_indices = set(range(min_index, max_index + 1))
        observed_index_set = set(observed_indices)

        missing_chunk_indices = sorted(
            expected_indices - observed_index_set
        )

        ordered_rows = [row for _, row in indexed_rows]

        if unindexed_rows:
            ordered_rows.extend(unindexed_rows)
            ordering_guarantee = "mixed_indexed_and_unindexed"
        else:
            ordering_guarantee = "explicit_sequence_index"

        expected_chunks = (
            len(observed_indices) + len(missing_chunk_indices)
        )

    else:
        ordered_rows = matching_rows
        expected_chunks = len(matching_rows)
        missing_chunk_indices = []
        ordering_guarantee = "unindexed_legacy_sequence_limitation"

    fetched_chunks = len(ordered_rows)

    reconstructed_parts = []

    for row in ordered_rows:
        chunk_text = str(row.get("text", "") or "").strip()

        if not chunk_text:
            continue

        if not reconstructed_parts:
            reconstructed_parts.append(chunk_text)
            continue

        previous_text = reconstructed_parts[-1]

        previous_words = previous_text.split()
        current_words = chunk_text.split()

        overlap_len = 0

        max_check = min(
            15,
            len(previous_words),
            len(current_words)
        )

        for n in range(max_check, 0, -1):
            if previous_words[-n:] == current_words[:n]:
                overlap_len = n
                break

        if overlap_len > 0:
            trimmed_current = " ".join(
                current_words[overlap_len:]
            )

            if trimmed_current:
                reconstructed_parts.append(trimmed_current)
        else:
            reconstructed_parts.append(chunk_text)

    reconstructed_text = "\n".join(reconstructed_parts)

    reconstructed_chunks = len(reconstructed_parts)

    missing_chunks = len(missing_chunk_indices)

    validation_metrics = {
        "expected_chunks": expected_chunks,
        "fetched_chunks": fetched_chunks,
        "missing_chunks": missing_chunks,
        "duplicate_chunks": duplicate_chunks,
        "reconstructed_chunks": reconstructed_chunks,
        "is_complete": (
            missing_chunks == 0
            and duplicate_chunks == 0
            and fetched_chunks > 0
            and reconstructed_chunks > 0
        ),
        "ordering_guarantee": ordering_guarantee,
        "missing_chunk_indices": missing_chunk_indices
    }

    return (
        reconstructed_text,
        expected_chunks,
        validation_metrics
    )


def get_stored_ocr_text_for_image(source_path):
	"""
	Retrieves stored OCR text for an image by searching both LanceDB image_documents and documents tables.
	Strips VLM refusal header lines (e.g. 'I cannot provide an analysis of a video...') and extracts pure OCR text.
	Returns tuple: (ocr_text, chunk_count)
	"""
	if not source_path:
		return "", 0

	target_base = os.path.basename(source_path).lower()
	matching_rows = []

	# 1. Search image_table
	image_table = get_image_table()
	if image_table is not None and image_table.count_rows() > 0:
		df_img = image_table.to_pandas()
		if not df_img.empty:
			for i in range(len(df_img)):
				row = df_img.iloc[i]
				r_path = str(row.get("path", ""))
				r_base = os.path.basename(r_path).lower()
				if r_base == target_base or source_path.lower() in r_path.lower():
					matching_rows.append(row)

	# 2. Search main documents table for image text chunks
	doc_table = get_table()
	if doc_table is not None and doc_table.count_rows() > 0:
		df_doc = doc_table.to_pandas()
		if not df_doc.empty:
			for i in range(len(df_doc)):
				row = df_doc.iloc[i]
				r_path = str(row.get("path", ""))
				r_base = os.path.basename(r_path).lower()
				if r_base == target_base or source_path.lower() in r_path.lower():
					matching_rows.append(row)

	if not matching_rows:
		return "", 0

	ocr_lines = []
	seen_lines = set()

	for row in matching_rows:
		raw_text = str(row.get("text", ""))

		# Split into lines and filter refusal strings and header lines
		lines = raw_text.split("\n")
		for line in lines:
			stripped = line.strip()
			if not stripped:
				continue

			# Skip header prefixes and VLM refusal sentences
			if stripped.startswith("File:") or stripped.startswith("Visual Summary") or stripped.startswith("Faces Detected:"):
				continue
			if any(refusal in stripped.lower() for refusal in [
				"cannot provide an analysis", "upload the video", "not provided a specific image",
				"cannot summarize", "without being able to see"
			]):
				continue
			if stripped == "OCR Text:":
				continue

			if stripped not in seen_lines:
				seen_lines.add(stripped)
				ocr_lines.append(stripped)

	full_ocr = "\n".join(ocr_lines).strip()
	return full_ocr, len(matching_rows)


# Backward compatibility alias for full content retrieval
get_full_transcript_for_source = get_full_content_for_source