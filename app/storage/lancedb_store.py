import lancedb
import os

DB_PATH='database'

#create database folder if it doesn't exists
os.makedirs(DB_PATH,exist_ok=True)

#Connect to lancedb
db=lancedb.connect(DB_PATH)

TABLE_NAME="documents"

#create table
def get_table():
	#open the table if exists, otherwise create
	table_names=db.table_names()

	if TABLE_NAME in table_names:
		return db.open_table(TABLE_NAME)

	return db.create_table(
	TABLE_NAME, data=[]
	)

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
			"vector":embedding
		}
	])

	print(f"Stored Chunk: {chunk_id}")

#Show all records
def show_all():
	return table.to_list()

#Total documents
def total_chunks():
	return table.count_rows()


if __name__=="__main__":
	print(show_all())
