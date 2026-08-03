import lancedb
import os
import pyarrow as pa

DB_PATH='database'
TABLE_NAME="documents"


#create database folder if it doesn't exists
os.makedirs(DB_PATH,exist_ok=True)

#Connect to lancedb
db=lancedb.connect(DB_PATH)


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
		pa.field("embedding", pa.list_(pa.float32(),384)),
		])

	return db.create_table(
		TABLE_NAME, schema=schema)

table=get_table()		

#Store one chunk
def store_chunk(chunk_id,path, file_type, text, embedding):
	#Store one document chunk into LanceDB
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

#Show all records
def show_all():
	return table.to_pandas()

#Total documents
def total_chunks():
	return table.count_rows()


if __name__=="__main__":
	print(table.schema)
	print("Total chunks: ",total_chunks())
