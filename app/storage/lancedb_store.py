import lancedb
import os
import pyarrow as pa
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "database")

TABLE_NAME="documents"
HASH_TABLE_NAME="processed_files"
IMAGE_TABLE_NAME = "image_documents"
FACE_TABLE_NAME = "face_embeddings"


#create database folder if it doesn't exists
os.makedirs(DB_PATH,exist_ok=True)

#Connect to lancedb
db=lancedb.connect(DB_PATH)

#-------Document Table--------
#create table
def get_table():
	tables = db.list_tables().tables
	if TABLE_NAME in tables:
		return db.open_table(TABLE_NAME)


	schema=pa.schema([
		pa.field("chunk_id", pa.string()),
		pa.field("path", pa.string()),
		pa.field("file_type", pa.string()),
		pa.field("text", pa.string()),
		
		#Text embedding
		pa.field("embedding", pa.list_(pa.float32(),384)),
		])


	return db.create_table(
		TABLE_NAME, schema=schema)


#-----Image Document Table------
def get_image_table():
	if IMAGE_TABLE_NAME in db.list_tables().tables:
		return db.open_table(IMAGE_TABLE_NAME)

	schema=pa.schema([
			pa.field("chunk_id",pa.string()),
			pa.field("path", pa.string()),
			pa.field("file_type", pa.string()),
			pa.field("text",pa.string()),

			#CLIP image embedding=512 dimensions
			pa.field("image_embedding", pa.list_(pa.float32(), 512)
				),
			])
	return db.create_table(
		IMAGE_TABLE_NAME,
		schema=schema
	)					


## -----Face Embeddings Table------
def get_face_table():
	if FACE_TABLE_NAME in db.list_tables().tables:
		ftable = db.open_table(FACE_TABLE_NAME)
		if "status" not in ftable.schema.names or "identity_source" not in ftable.schema.names:
			try:
				ftable.add_columns({"status": "'unknown'", "identity_source": "'none'"})
			except Exception:
				pass
		return ftable

	schema = pa.schema([
		pa.field("face_id", pa.string()),
		pa.field("image_path", pa.string()),
		pa.field("bbox", pa.list_(pa.int32(), 4)),
		pa.field("person_name", pa.string()),
		pa.field("status", pa.string()),
		pa.field("identity_source", pa.string()),
		pa.field("face_embedding", pa.list_(pa.float32(), 512)),
		pa.field("created_at", pa.string())
	])
	return db.create_table(
		FACE_TABLE_NAME,
		schema=schema
	)


#Processed file hash table
def get_hash_table():
	if HASH_TABLE_NAME in db.list_tables().tables:
		return db.open_table(HASH_TABLE_NAME)

	schema=pa.schema([
		pa.field("file_hash", pa.string()),
		pa.field("path", pa.string()),
		pa.field("created_at", pa.string()),
		])
	return db.create_table(HASH_TABLE_NAME, schema=schema)		

FEEDBACK_TABLE_NAME = "rag_feedback"

# -----Feedback / Learning Memory Table------
def get_feedback_table():
	if FEEDBACK_TABLE_NAME in db.list_tables().tables:
		return db.open_table(FEEDBACK_TABLE_NAME)

	schema = pa.schema([
		pa.field("id", pa.string()),
		pa.field("timestamp", pa.string()),
		pa.field("original_query", pa.string()),
		pa.field("wrong_answer", pa.string()),
		pa.field("wrong_source", pa.string()),
		pa.field("correct_source", pa.string()),
		pa.field("corrected_answer", pa.string()),
		pa.field("correction_text", pa.string()),
		pa.field("modality", pa.string()),
		pa.field("rejected_sources", pa.string()),
		pa.field("confidence", pa.float32()),
		pa.field("feedback_type", pa.string()),
		pa.field("embedding", pa.list_(pa.float32(), 384)),
	])
	return db.create_table(
		FEEDBACK_TABLE_NAME,
		schema=schema
	)

table=get_table()
hash_table=get_hash_table()
image_table = get_image_table()
face_table = get_face_table()
feedback_table = get_feedback_table()


def store_feedback(
	original_query,
	wrong_answer="",
	wrong_source="",
	correct_source="",
	corrected_answer="",
	correction_text="",
	modality="all",
	rejected_sources="",
	confidence=1.0,
	feedback_type="source_correction"
):
	"""
	Stores user correction / feedback into the rag_feedback LanceDB table.
	Computes text embedding for original_query so future queries can perform vector similarity search.
	"""
	import uuid
	from app.embeddings.embedding_router import generate_embedding

	f_table = get_feedback_table()
	record_id = str(uuid.uuid4())
	timestamp_str = str(datetime.now())

	# Generate text embedding for original_query
	query_embedding = generate_embedding("text", original_query)

	feedback_record = {
		"id": record_id,
		"timestamp": timestamp_str,
		"original_query": original_query,
		"wrong_answer": wrong_answer,
		"wrong_source": wrong_source,
		"correct_source": correct_source,
		"corrected_answer": corrected_answer,
		"correction_text": correction_text,
		"modality": modality,
		"rejected_sources": rejected_sources,
		"confidence": float(confidence),
		"feedback_type": feedback_type,
		"embedding": query_embedding,
	}

	f_table.add([feedback_record])
	print(f"Stored Feedback Record: {record_id} for query '{original_query}'")
	return feedback_record


def search_feedback(query, query_modality="all", source_hint=None, limit=5):
	"""
	Searches stored feedback entries in rag_feedback table matching the given query.
	Applies strict modality, source_hint, and distance threshold filtering to prevent cross-modality feedback leak.
	"""
	f_table = get_feedback_table()
	if f_table is None:
		return []

	# Check count
	if f_table.count_rows() == 0:
		return []

	from app.embeddings.embedding_router import generate_embedding
	query_vector = generate_embedding("text", query)

	raw_results = f_table.search(query_vector).metric("cosine").limit(limit * 3).to_list()
	filtered = []

	for fb in raw_results:
		fb_modality = str(fb.get("modality", "all")).lower()
		fb_dist = float(fb.get("_distance", 1.0))

		# 1. Strict Modality Filter: Audio feedback must NOT affect Image queries
		if query_modality != "all" and fb_modality != "all" and fb_modality != str(query_modality).lower():
			continue

		# 2. Distance Threshold: Require vector closeness (distance <= 0.50)
		if fb_dist > 0.50:
			continue

		# 3. Source Filter: If source_hint is specified, ensure correct_source or wrong_source matches
		if source_hint:
			sh_lower = str(source_hint).lower()
			fb_c_src = str(fb.get("correct_source", "")).lower()
			fb_w_src = str(fb.get("wrong_source", "")).lower()
			if fb_c_src and sh_lower not in fb_c_src and fb_w_src and sh_lower not in fb_w_src:
				continue

		filtered.append(fb)
		if len(filtered) >= limit:
			break

	return filtered


#File hash function
def is_file_processed(file_hash):
	result=hash_table.search().where(f"file_hash='{file_hash}'").limit(1).to_list()
	return len(result) > 0

def save_file_hash(file_hash, path):
	hash_table.add([
		{
			"file_hash":file_hash,
			"path":path,
			"created_at":str(datetime.now())
		}	
	])	
	print(f"Saved hash: {file_hash}")

#Store one chunk
def store_chunk(chunk_id, path, file_type, text, embedding):
	#Store one document chunk into LanceDB.
	
	table.add([
		{
			"chunk_id":chunk_id,
			"path":path,
			"file_type":file_type,
			"text":text,
			"embedding":embedding
		}
	])

	print(f"Stored Chunk: {chunk_id}")

#Image chunk store
def store_image_chunk(chunk_id, path, file_type, text, image_embedding):
	#Store image information and its 512-D CLIP embedding in the separate image_documents table.
	image_table.add([
		{
			"chunk_id":chunk_id,
			"path":path,
			"file_type":file_type,
			"text":text,
			"image_embedding":image_embedding,
		}
	])	

	print(f"Stored Image Chunks: {chunk_id}")

#Store face record
def store_face_record(face_id, image_path, bbox, face_embedding, person_name="unknown", status="unknown", identity_source="none"):
	"""
	Stores one detected face and its 512-D face embedding vector in the face_embeddings table.
	Enforces independent tracking of person_name, status ('known'|'unknown'), and identity_source.
	"""
	import uuid
	ftable = get_face_table()

	final_face_id = face_id if face_id else f"face_{uuid.uuid4()}"
	final_status = status
	if person_name and person_name.lower() != "unknown":
		final_status = "known"

	record = {
		"face_id": final_face_id,
		"image_path": os.path.abspath(image_path),
		"bbox": bbox,
		"person_name": person_name,
		"status": final_status,
		"identity_source": identity_source,
		"face_embedding": face_embedding,
		"created_at": str(datetime.now())
	}
	ftable.add([record])
	print(f"Stored Face Record: {final_face_id} ({person_name}, status={final_status}, src={identity_source})")
	return record


def compute_bbox_iou(box1, box2):
	x1, y1, w1, h1 = box1
	x2, y2, w2, h2 = box2
	xi1 = max(x1, x2)
	yi1 = max(y1, y2)
	xi2 = min(x1 + w1, x2 + w2)
	yi2 = min(y1 + h1, y2 + h2)
	inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
	box1_area = w1 * h1
	box2_area = w2 * h2
	union_area = box1_area + box2_area - inter_area
	if union_area == 0:
		return 0.0
	return inter_area / union_area


def deduplicate_and_store_faces(image_path, detected_faces):
	"""
	Deduplicates faces for image_path to prevent duplicate accumulation in LanceDB.
	Matches existing records by image_path and bounding box IoU >= 0.4.
	"""
	import uuid
	ftable = get_face_table()
	if ftable is None:
		return []

	df = ftable.to_pandas()
	abs_path = os.path.abspath(image_path)

	existing_rows = []
	if df is not None and not df.empty and "image_path" in df.columns:
		existing_rows = df[df["image_path"] == abs_path].to_dict(orient="records")

	stored_records = []
	for face in detected_faces:
		bbox = face["bbox"]
		embedding = face["embedding"]

		matched_existing = None
		for ex in existing_rows:
			ex_bbox = ex.get("bbox")
			if ex_bbox is not None:
				iou = compute_bbox_iou(bbox, ex_bbox)
				if iou >= 0.4:
					matched_existing = ex
					break

		if matched_existing:
			stored_records.append(matched_existing)
		else:
			face_id = f"face_{uuid.uuid4()}"
			rec = store_face_record(
				face_id=face_id,
				image_path=abs_path,
				bbox=bbox,
				face_embedding=embedding,
				person_name="unknown",
				status="unknown",
				identity_source="none"
			)
			stored_records.append(rec)

	return stored_records


def update_face_record_identity(face_id, person_name, identity_source="user_registration"):
	"""
	Updates the identity of an existing persistent face record in LanceDB while preserving
	its exact face_id, face_embedding, bbox, and image_path.
	"""
	ftable = get_face_table()
	if ftable is None:
		return None

	df = ftable.to_pandas()
	if df.empty or "face_id" not in df.columns:
		return None

	matching = df[df["face_id"] == face_id]
	if matching.empty:
		return None

	row = matching.iloc[0].to_dict()
	# Delete existing row by face_id
	try:
		ftable.delete(f"face_id = '{face_id}'")
	except Exception:
		pass

	row["person_name"] = person_name
	row["status"] = "known" if person_name and person_name.lower() != "unknown" else "unknown"
	row["identity_source"] = identity_source

	ftable.add([row])
	print(f"Updated Face Record Identity: {face_id} -> '{person_name}' ({identity_source})")
	return row


#Show all records
def show_all():
	return table.to_pandas()

def show_processed_files():
	return hash_table.to_pandas()	

#Total documents
def total_chunks():
	return table.count_rows()

def total_files():
	return hash_table.count_rows()

def total_faces():
	ftable = get_face_table()
	return ftable.count_rows()

if __name__=="__main__":
	print("Document schema: ")
	print(table.schema)

	print("\nHash Table schema: ")
	print(hash_table.schema)

	print("Total chunks: ",total_chunks())
	print("Total processed files: ", total_files())