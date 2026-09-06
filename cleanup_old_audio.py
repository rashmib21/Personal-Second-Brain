import datetime
import os
import shutil
import lancedb

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database")
WATCHED_FOLDER = os.path.join(BASE_DIR, "watched_folder")

# Target audio files to clean up
TARGET_FILENAMES = ["audio.mpeg", "test_audio.wav", "test_audio.mpeg"]


def backup_database():
    """Create a backup copy of the database folder before deletion."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DB_PATH}_backup_audio_cleanup_{timestamp}"
    if os.path.exists(DB_PATH):
        shutil.copytree(DB_PATH, backup_path)
        print(f"[BACKUP SUCCESS] Database backup created at: {backup_path}")
        return backup_path
    else:
        print("[BACKUP WARNING] Database path does not exist.")
        return None


def cleanup_old_audio():
    """Remove old audio records from LanceDB and delete files from watched_folder."""
    # 1. Connect to LanceDB
    db = lancedb.connect(DB_PATH)
    doc_table = db.open_table("documents")
    hash_table = db.open_table("processed_files")

    initial_doc_count = doc_table.count_rows()
    initial_hash_count = hash_table.count_rows()

    print(f"\n[INITIAL STATE] 'documents' total rows: {initial_doc_count}")
    print(f"[INITIAL STATE] 'processed_files' total rows: {initial_hash_count}")

    # Build target paths (absolute and relative) to match in LanceDB
    target_paths = []
    for fname in TARGET_FILENAMES:
        target_paths.append(os.path.join(WATCHED_FOLDER, fname))
        target_paths.append(f"./watched_folder/{fname}")
        target_paths.append(fname)

    # 2. Delete matching records from 'documents' table
    print("\n--- Cleaning LanceDB 'documents' Table ---")
    deleted_docs_count = 0
    for path in target_paths:
        filter_expr = f'path = "{path}"'
        matches = doc_table.search().where(filter_expr).to_pandas()
        if len(matches) > 0:
            print(f"Deleting {len(matches)} record(s) from 'documents' for path: {path}")
            doc_table.delete(filter_expr)
            deleted_docs_count += len(matches)

    # 3. Delete matching records from 'processed_files' table
    print("\n--- Cleaning LanceDB 'processed_files' Table ---")
    deleted_hashes_count = 0
    for path in target_paths:
        filter_expr = f'path = "{path}"'
        matches = hash_table.search().where(filter_expr).to_pandas()
        if len(matches) > 0:
            print(f"Deleting {len(matches)} record(s) from 'processed_files' for path: {path}")
            hash_table.delete(filter_expr)
            deleted_hashes_count += len(matches)

    # 4. Remove physical audio files from watched_folder ONLY
    print("\n--- Removing Audio Files from watched_folder ---")
    for fname in TARGET_FILENAMES:
        file_path = os.path.join(WATCHED_FOLDER, fname)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted physical file: {file_path}")
        else:
            print(f"Physical file not found (already deleted): {file_path}")

    # 5. Summary and Verification
    final_doc_count = doc_table.count_rows()
    final_hash_count = hash_table.count_rows()

    print("\n--- Summary & Verification ---")
    print(f"'documents' count: {initial_doc_count} -> {final_doc_count} (Removed {deleted_docs_count} rows)")
    print(f"'processed_files' count: {initial_hash_count} -> {final_hash_count} (Removed {deleted_hashes_count} rows)")

    # Assertions for verification
    for fname in TARGET_FILENAMES:
        file_path = os.path.join(WATCHED_FOLDER, fname)
        assert not os.path.exists(file_path), f"Verification failed: File {file_path} still exists!"
        for p in [file_path, f"./watched_folder/{fname}", fname]:
            rem_docs = doc_table.search().where(f'path = "{p}"').to_pandas()
            assert len(rem_docs) == 0, f"Verification failed: Document record for {p} still exists!"
            rem_hashes = hash_table.search().where(f'path = "{p}"').to_pandas()
            assert len(rem_hashes) == 0, f"Verification failed: Hash record for {p} still exists!"

    print("\n[VERIFICATION PASSED] All target old audio files and processing states were successfully removed.")


if __name__ == "__main__":
    backup_database()
    cleanup_old_audio()
