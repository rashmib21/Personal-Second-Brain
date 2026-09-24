#!/usr/bin/env python3
"""
One-time migration:
1. Read spreadsheet files from watched_folder.
2. Store every spreadsheet row structurally in LanceDB table `spreadsheet_rows`.
3. Delete ONLY old spreadsheet rows from `documents`.
4. Keep PDFs/DOCX/TXT/images untouched.

Run first:
    python migrate_spreadsheets.py --dry-run

Then:
    python migrate_spreadsheets.py --migrate
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import lancedb
import pandas as pd
import pyarrow as pa
from app.rag.spreadsheet_query import load_spreadsheet_sheets

# Project root = directory containing watched_folder/database
PROJECT_ROOT = Path(__file__).resolve().parent
WATCHED_FOLDER = PROJECT_ROOT / "watched_folder"
DATABASE = PROJECT_ROOT / "database"

SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv", ".ods"}
SPREADSHEET_TABLE = "spreadsheet_rows"
DOCUMENTS_TABLE = "documents"


def clean_value(value):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_workbook(path):
    """Same loader as the query engine: real header row, no banner/legend/section rows."""
    if path.suffix.lower() == ".csv":
        return {"Sheet1": pd.read_csv(path, dtype=object)}
    return {name: d["df"] for name, d in load_spreadsheet_sheets(str(path)).items()}


def make_search_text(source_file, sheet_name, row_number, row_data):
    parts = [
        f"File: {source_file}",
        f"Sheet: {sheet_name}",
        f"Row: {row_number}",
    ]

    for key, value in row_data.items():
        if value is not None and str(value).strip():
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


def collect_rows():
    records = []
    spreadsheets = []

    for path in sorted(WATCHED_FOLDER.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SPREADSHEET_EXTENSIONS:
            continue

        spreadsheets.append(path)
        sha = file_hash(path)
        workbook = load_workbook(path)

        for sheet_name, df in workbook.items():
            if df is None:
                continue

            # Preserve original Excel headers exactly.
            headers = [str(c) for c in df.columns]

            for excel_index, (_, row) in enumerate(df.iterrows(), start=2):
                row_data = {
                    header: clean_value(row.iloc[i])
                    for i, header in enumerate(headers)
                }

                # Skip completely empty rows.
                if not any(
                    value is not None and str(value).strip()
                    for value in row_data.values()
                ):
                    continue

                source_file = path.name
                search_text = make_search_text(
                    source_file,
                    str(sheet_name),
                    excel_index,
                    row_data,
                )

                records.append(
                    {
                        "chunk_id": hashlib.sha256(
                            f"{sha}:{sheet_name}:{excel_index}".encode("utf-8")
                        ).hexdigest(),
                        "source_file": source_file,
                        "source_path": str(path.resolve()),
                        "file_hash": sha,
                        "sheet_name": str(sheet_name),
                        "row_number": int(excel_index),
                        "row_data": json.dumps(
                            row_data,
                            ensure_ascii=False,
                            default=str,
                        ),
                        "search_text": search_text,
                    }
                )

    return spreadsheets, records


def get_documents_table(db):
    if DOCUMENTS_TABLE not in db.list_tables().tables:
        return None
    return db.open_table(DOCUMENTS_TABLE)


def get_or_create_spreadsheet_table(db):
    if SPREADSHEET_TABLE in db.list_tables().tables:
        return db.open_table(SPREADSHEET_TABLE)

    schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("sheet_name", pa.string()),
            pa.field("row_number", pa.int64()),
            pa.field("row_data", pa.string()),
            pa.field("search_text", pa.string()),
        ]
    )

    return db.create_table(SPREADSHEET_TABLE, schema=schema)


def migrate():
    db = lancedb.connect(str(DATABASE))
    spreadsheets, records = collect_rows()

    print(f"\nSpreadsheet files found: {len(spreadsheets)}")
    for path in spreadsheets:
        print(f"  - {path.name}")

    print(f"\nStructured rows to insert: {len(records)}")

    if not records:
        print("Nothing to migrate.")
        return

    # First create/insert the structured representation.
    table = get_or_create_spreadsheet_table(db)

    source_paths = sorted({r["source_path"] for r in records})

    # Remove previous structured rows for exactly these files.
    for source_path in source_paths:
        escaped = source_path.replace("'", "''")
        try:
            table.delete(f"source_path = '{escaped}'")
        except Exception as exc:
            print(f"Warning: could not delete old structured rows for {source_path}: {exc}")

    table.add(records)
    print(f"Inserted {len(records)} rows into `{SPREADSHEET_TABLE}`.")

    # Only after successful structured insertion, remove old spreadsheet
    # representations from the generic documents table.
    documents = get_documents_table(db)

    if documents is None:
        print("No documents table found; nothing to remove there.")
        return

    for source_path in source_paths:
        escaped = source_path.replace("'", "''")
        try:
            documents.delete(f"path = '{escaped}'")
            print(f"Removed old document chunks: {Path(source_path).name}")
        except Exception as exc:
            print(
                f"WARNING: structured rows are saved, but old chunks could "
                f"not be removed for {source_path}: {exc}"
            )

    print("\nMigration complete.")
    print("PDF/DOCX/TXT/image rows in `documents` were not targeted.")


def dry_run():
    spreadsheets, records = collect_rows()

    print(f"\nWatched folder: {WATCHED_FOLDER}")
    print(f"Spreadsheet files: {len(spreadsheets)}")

    for path in spreadsheets:
        print(f"  - {path.name}")

    print(f"\nRows that will become structured records: {len(records)}")

    for record in records[:5]:
        print("\n--- sample ---")
        print("file:", record["source_file"])
        print("sheet:", record["sheet_name"])
        print("row:", record["row_number"])
        print("row_data:", record["row_data"])

    if len(records) > 5:
        print(f"\n... {len(records) - 5} more rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect what will be migrated without modifying LanceDB.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Create/update spreadsheet_rows and remove old spreadsheet chunks.",
    )
    args = parser.parse_args()

    if args.dry_run == args.migrate:
        parser.error("Use exactly one of --dry-run or --migrate.")

    if args.dry_run:
        dry_run()
    else:
        migrate()


if __name__ == "__main__":
    main()
