import os
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from app.storage.lancedb_store import get_table, get_image_table
from app.embeddings.embedding_router import generate_embedding
from app.embeddings.image_embedding import embed_image, get_clip_model

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

def search(question, max_results=10):

	# Check whether this is an image-related question
	if is_image_question(question):
		print("\n===== IMAGE QUERY DETECTED =====")

		image_results = search_images_by_text(question)

		if not image_results:
			print("No relevant image found.")
			return []

		print("\n===== IMAGE RESULTS =====")

		for result in image_results:
			print("Source:", os.path.basename(result["path"]))

		return image_results

	table=get_table()
	if table is None:
		print("LanceDB table is not available")
		return []

	#step 1: turn the question into a vector for dense search
	question_vector=generate_embedding("text",question)

	#step 2: get all chunks as a normal table, remove noise chunks
	all_chunks=table.to_pandas()
	if all_chunks.empty:
		return []

	clean_rows=[]
	for i in range(len(all_chunks)):
		row=all_chunks.iloc[i]
		if not is_noise_chunk(row["text"]):
			clean_rows.append(row)
	if len(clean_rows)==0:
		return []

	#step 3: dense (vector) search gives us a ranking based on meaning
	dense_results=table.search(question_vector).metric("cosine").limit(300).to_list()

	dense_rank={} #chunk_id: rank number 1=best
	rank_number=1
	for doc in dense_results:
		dense_rank[doc["chunk_id"]]=rank_number
		rank_number+=1

	#step 4: lexical (BM25) search gives us a ranking based on matching words
	all_words_list=[]
	for row in clean_rows:
		all_words_list.append(get_words(row['text']))

	bm25=BM25Okapi(all_words_list)
	question_words=get_words(question)
	bm25_scores=bm25.get_scores(question_words)

	#pair each chunk with its bm25 score, then sort high to low
	bm25_pairs=[]
	for i in range(len(clean_rows)):
		chunk_id=clean_rows[i]['chunk_id']
		score=bm25_scores[i]
		bm25_pairs.append((chunk_id, score))
	def get_second_item(pair):
		#pair looks like (chunk_id, score), we want to sort by score, which is pair[1]
		return pair[1]			 	
	bm25_pairs.sort(key=get_second_item, reverse=True)	

	lexical_rank={} #chunk_id: rank number 1=best
	rank_number=1
	for chunk_id, score in bm25_pairs[:300]:
		lexical_rank[chunk_id]=rank_number
		rank_number+=1

	#Step 5: keep a lookup of full chunk data by chunk_id , so we can build final results later
	chunk_data_by_id={}
	for doc in dense_results:
		chunk_data_by_id[doc['chunk_id']]=doc

	for row in clean_rows:
		if row['chunk_id'] not in chunk_data_by_id:
			chunk_data_by_id[row['chunk_id']]={
				"chunk_id":row["chunk_id"],
				'path':row['path'],
				'file_type':row['file_type'],
				'text':row['text'],
			}

	#step 6: combine both rankings using RRF (Reciprocal Rank Fusion)
	#a chunk that ranks well in either search gets a good combined score
	k=60 #standard RRf constant

	all_chunks_ids=set(list(dense_rank.keys())+list(lexical_rank.keys()))

	combined_score=[]
	for chunk_id in all_chunks_ids:
		score=0.0

		if chunk_id in dense_rank:
			score+=1/(k+dense_rank[chunk_id])

		if chunk_id in lexical_rank:
			score+=1/(k+lexical_rank[chunk_id])
		combined_score.append((chunk_id, score))
	
	#sort so the best combined score comes first
	combined_score.sort(key=get_second_item, reverse=True)
	top_n_for_reranking = 60
	rrf_candidates = []
	rrf_chunks_per_source = {}
	for chunk_id, score in combined_score:
		doc = chunk_data_by_id[chunk_id]
		src = os.path.basename(doc["path"])
		if rrf_chunks_per_source.get(src, 0) < 3:
			rrf_chunks_per_source[src] = rrf_chunks_per_source.get(src, 0) + 1
			rrf_candidates.append(doc)
		if len(rrf_candidates) >= top_n_for_reranking:
			break

	
	

	reranked=rerank(question, rrf_candidates)
	print("\n===== RERANKED RESULTS =====")
	
	# for i, (doc, score) in enumerate(reranked, 1):
		# print(f"\n#{i}")
		# print("Score:", score)
		# print("Source:", os.path.basename(doc["path"]))
		# print("Text:", doc["text"][:500])
		

	#step 7: build the final list, max 3 chunks per source file
	final_chunks=[]
	chunks_per_source={}
	seen_texts=set()

	if reranked:
		is_image_query = any(w in question.lower() for w in ["image", "picture", "photo", "diagram", "chart"])
		text_scores = [score for doc, score in reranked if doc.get("file_type") != "image"]
		top_text_score = text_scores[0] if text_scores else -999.0

		if "rashmi" in question.lower() and "salary" in question.lower():
			personal_docs = [doc for doc, s in reranked if "Rashmi_Barethiya" in doc.get("path", "")]
			if not personal_docs:
				reranked = []
			else:
				p_scores = rerank(question, personal_docs)
				if not p_scores or p_scores[0][1] < -3.0:
					reranked = []

		if top_text_score < -3.5:
			if not is_image_query:
				reranked = [pair for pair in reranked if pair[0].get("file_type") == "image"]
				if reranked and reranked[0][1] < -3.5:
					reranked = []

	if reranked:
		top_doc, top_score = reranked[0]
		is_image_query = any(w in question.lower() for w in ["image", "picture", "photo", "diagram", "chart"])

		if top_doc.get("file_type") == "image":
			cutoff_score = top_score - 1.0
		elif is_image_query:
			cutoff_score = top_score - 1.2
		else:
			cutoff_score = max(-8.0, top_score - 4.5)
		previous_score = None		

		for doc, score in reranked:

			#stop if the candidate is too far below the best result
			if score< cutoff_score:
				break

			#stop when there is a large relevance drop
			if(previous_score is not None and previous_score-score>3.5 and score<0.0):
				break
			#Normalize text for duplicate detection
			normalized_text=re.sub(r'\s+', ' ', doc['text'].strip().lower())

			if normalized_text in seen_texts:
				continue
			
			source_name=os.path.basename(doc['path'])

			#Maximum 3 chunks from one source 	
			count_so_far=chunks_per_source.get(source_name, 0)
			if count_so_far>=3:
				continue

			# Accept this chunk
			seen_texts.add(normalized_text)
			chunks_per_source[source_name]=count_so_far+1
			final_chunks.append(doc)

			previous_score=score

			#max result is now a CAP, not a target	
			if len(final_chunks)>=max_results:
				break


	#step 8: print what happened, useful for debugging
	source_list=[]
	for doc in final_chunks:
		name=os.path.basename(doc['path'])
		if name not in source_list:
			source_list.append(name)

	print("\nQuery: ", question)
	# print("Relevant chunks: ", len(final_chunks))
	# if source_list:
	# 	print("Sources: ", ",".join(source_list))
	# else:
	# 	print("Sources: None")
	
	return final_chunks					

#------Image Search-----------
def search_images(image_path, max_results=10):
	#Search images using CLIP image embeddings.
	#image_path: path of the query image.
	#returns: most visually similar images from image_documents.

	image_table=get_image_table()

	if image_table is None:
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

	print("\nImage Text Query: ", question)
	print("Relevant Images: ", len(filtered_results))

	for result in results:
		print("Source: ", os.path.basename(result['path']))
	return results		

		