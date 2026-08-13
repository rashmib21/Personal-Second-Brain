import gzip
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile

logger = logging.getLogger(__name__)

def extract_archive(path):
    """
    Extracts and parses supported files from archive files (.zip, .tar, .tar.gz, .tgz, .gz).
    Extracted files are processed internally in a temporary directory and cleaned up via try/finally.
    Returns structured result dict.
    """
    from app.extractors.router import extract_file

    ext = os.path.splitext(path)[1].lower()

    if ext in [".rar", ".7z"]:
        logger.warning(f"Unsupported archive format ({ext}): {path}")
        return {
            "text": "",
            "status": "UNSUPPORTED_FORMAT",
            "error": f"Archive format {ext} is not supported"
        }

    temp_dir = tempfile.mkdtemp(prefix="second_brain_archive_")
    extracted_texts = []

    try:
        # Case 1: ZIP Archive
        if ext == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    for member in archive.infolist():
                        # Prevent Zip Slip / path traversal
                        target_path = os.path.abspath(os.path.join(temp_dir, member.filename))
                        if not target_path.startswith(os.path.abspath(temp_dir)):
                            logger.warning(f"Skipping dangerous member in zip {member.filename}")
                            continue

                        if member.is_dir():
                            os.makedirs(target_path, exist_ok=True)
                            continue

                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with archive.open(member) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                        res = extract_file(target_path)
                        if isinstance(res, dict) and res.get("status") == "SUCCESS" and res.get("text"):
                            child_name = member.filename
                            extracted_texts.append(f"--- Archive Member: {child_name} ---\n{res['text']}")

            except zipfile.BadZipFile as e:
                logger.error(f"Corrupt ZIP file {path}: {e}")
                return {"text": "", "status": "CORRUPT_FILE", "error": str(e)}

        # Case 2: TAR / TAR.GZ / TGZ Archive
        elif ext in [".tar", ".tgz"] or (ext == ".gz" and path.lower().endswith(".tar.gz")):
            try:
                with tarfile.open(path, "r:*") as tar:
                    for member in tar.getmembers():
                        target_path = os.path.abspath(os.path.join(temp_dir, member.name))
                        if not target_path.startswith(os.path.abspath(temp_dir)):
                            logger.warning(f"Skipping dangerous member in tar {member.name}")
                            continue

                        if member.isdir():
                            os.makedirs(target_path, exist_ok=True)
                            continue

                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(target_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                            res = extract_file(target_path)
                            if isinstance(res, dict) and res.get("status") == "SUCCESS" and res.get("text"):
                                extracted_texts.append(f"--- Archive Member: {member.name} ---\n{res['text']}")

            except Exception as e:
                logger.error(f"Corrupt TAR archive {path}: {e}")
                return {"text": "", "status": "CORRUPT_FILE", "error": str(e)}

        # Case 3: Standalone .gz File
        elif ext == ".gz":
            try:
                # Decompress to inner filename
                base_name = os.path.basename(path)
                inner_name = base_name[:-3] if base_name.lower().endswith(".gz") else "decompressed.txt"
                decompressed_path = os.path.join(temp_dir, inner_name)

                with gzip.open(path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

                res = extract_file(decompressed_path)
                if isinstance(res, dict) and res.get("status") == "SUCCESS" and res.get("text"):
                    extracted_texts.append(f"--- Decompressed GZ Content: {inner_name} ---\n{res['text']}")

            except Exception as e:
                logger.error(f"Failed to decompress GZ file {path}: {e}")
                return {"text": "", "status": "CORRUPT_FILE", "error": str(e)}

        else:
            return {"text": "", "status": "UNSUPPORTED_FORMAT", "error": f"Unsupported archive extension: {ext}"}

        full_text = "\n\n".join(extracted_texts).strip()
        if not full_text:
            return {
                "text": "",
                "status": "NO_TEXT_EXTRACTED",
                "error": "No extractable supported files found inside archive"
            }

        return {
            "text": full_text,
            "status": "SUCCESS",
            "error": None
        }

    except Exception as e:
        logger.error(f"Failed to process archive {path}: {e}")
        return {"text": "", "status": "PROCESSING_ERROR", "error": str(e)}

    finally:
        # Enforce temporary directory cleanup
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")