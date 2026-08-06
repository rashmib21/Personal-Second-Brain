import lancedb
import pyarrow as pa

db=lancedb.connect("lancedb_data")

schema=pa.schema([
	("id",pa.string()),
	("path",pa.string()),
	("file_type",pa.string()),
	("text",pa.string()),
	("embedding",pa.list_(pa.float32(),384))
])

	table=db.create_table(
		"documents",
		schema=schema,
	)

print(table)