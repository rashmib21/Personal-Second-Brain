"""
Backfill script to build database/source_catalog.json for all existing indexed files.
LanceDB is strictly READ-ONLY. No table recreation, migration, or deletion.
Uses local Ollama (ask_llama) for catalog generation.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill_catalog")

def run_backfill():
    from app.storage.lancedb_store import get_table, get_image_table
    from app.query.source_catalog import build_entry, save_entry, load_catalog
    from app.llm.ollama_client import ask_llama

    existing_catalog = load_catalog()
    existing_files = {e.get("file") for e in existing_catalog if e.get("file")}

    file_samples = {} # path: {"file_type": ..., "chunks": []}

    # 1. Read document table (read-only)
    table = get_table()
    if table is not None:
        try:
            df = table.to_pandas()
            if not df.empty:
                for _, row in df.iterrows():
                    p = str(row.get("path", ""))
                    ft = str(row.get("file_type", "document"))
                    txt = str(row.get("text", ""))
                    if p:
                        if p not in file_samples:
                            file_samples[p] = {"file_type": ft, "chunks": []}
                        file_samples[p]["chunks"].append(txt)
        except Exception as e:
            logger.warning(f"Error reading document table: {e}")

    # 2. Read image table (read-only)
    img_table = get_image_table()
    if img_table is not None:
        try:
            df_img = img_table.to_pandas()
            if not df_img.empty:
                for _, row in df_img.iterrows():
                    p = str(row.get("path", ""))
                    ft = "image"
                    txt = str(row.get("text", ""))
                    if p:
                        if p not in file_samples:
                            file_samples[p] = {"file_type": ft, "chunks": []}
                        file_samples[p]["chunks"].append(txt)
        except Exception as e:
            logger.warning(f"Error reading image table: {e}")

    processed_count = 0
    total_files = len(file_samples)
    logger.info(f"Found {total_files} unique files in LanceDB.")

    for path, data in file_samples.items():
        fname = os.path.basename(path)
        file_type = data["file_type"]
        sample_text = "\n".join(data["chunks"])[:2500]

        logger.info(f"Processing catalog entry ({processed_count + 1}/{total_files}): {fname}")
        try:
            entry = build_entry(path, sample_text, ask_llama, file_type)
            save_entry(entry)
            processed_count += 1
            logger.info(f"Successfully saved entry for {fname}")
        except Exception as exc:
            logger.error(f"Failed to generate entry for {fname}: {exc}")

    logger.info(f"Backfill complete! Generated catalog entries for {processed_count} files out of {total_files}.")
    return processed_count

if __name__ == "__main__":
    count = run_backfill()
    print(f"BACKFILL_COMPLETED_COUNT={count}")
