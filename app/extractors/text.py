import logging

logger = logging.getLogger(__name__)

def extract_txt(path):
    """
    Extract text from a plain text file (.txt, .md, .rtf).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()

        if not text.strip():
            return {
                "text": "",
                "status": "NO_TEXT_EXTRACTED",
                "error": None
            }

        return {
            "text": text,
            "status": "SUCCESS",
            "error": None
        }
    except Exception as e:
        logger.error(f"Error reading text file {path}: {e}")
        return {
            "text": "",
            "status": "CORRUPT_FILE",
            "error": str(e)
        }