from app.storage.lancedb_store import table
from tabulate import tabulate


def show_database():
    print("=" * 120)
    print("LanceDB Information")
    print("=" * 120)

    print(f"Table Name   : {table.name}")
    print(f"Total Chunks : {table.count_rows()}")

    print("\nSchema")
    print("-" * 120)
    print(table.schema)

    print("\nDatabase Contents")
    print("-" * 120)

    df = table.to_pandas()

    # Hide embedding column
    if "embedding" in df.columns:
        df = df.drop(columns=["embedding"])

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))


def show_chunks():
    df = table.to_pandas()

    print("\n\nChunk Details")
    print("=" * 120)

    for i, row in df.iterrows():
        print("=" * 120)
        print(f"Chunk {i+1}")
        print("=" * 120)

        print(f"Chunk ID : {row['chunk_id']}")
        print(f"Path     : {row['path']}")
        print(f"Type     : {row['file_type']}")

        print("\nText")
        print("-" * 120)
        print(row["text"])
        print()


if __name__ == "__main__":
    show_database()
    show_chunks()