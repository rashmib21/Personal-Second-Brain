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


#-----Face Embeddings Table------
def get_face_table():
	if FACE_TABLE_NAME in db.list_tables().tables:
		return db.open_table(FACE_TABLE_NAME)

	schema = pa.schema([
		pa.field("face_id", pa.string()),
		pa.field("image_path", pa.string()),
		pa.field("bbox", pa.list_(pa.int32(), 4)),
		pa.field("person_name", pa.string()),
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

table=get_table()
hash_table=get_hash_table()
image_table = get_image_table()
face_table = get_face_table()



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
def store_face_record(face_id, image_path, bbox, face_embedding, person_name="unknown"):
	#Store one detected face and its 512-D face embedding vector in the face_embeddings table.
	ftable = get_face_table()
	ftable.add([
		{
			"face_id": face_id,
			"image_path": image_path,
			"bbox": bbox,
			"person_name": person_name,
			"face_embedding": face_embedding,
			"created_at": str(datetime.now())
		}
	])
	print(f"Stored Face Record: {face_id} ({person_name})")

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