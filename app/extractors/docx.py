import logging
import os
from docx import Document

logger = logging.getLogger(__name__)

def extract_docx(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".doc":
        logger.warning(f"Legacy binary Word format (.doc) parsing attempted: {path}")

    try:
        document = Document(path)
        paragraphs = []
        for paragraph in document.paragraphs:
            p_text = paragraph.text.strip()
            if p_text:
                paragraphs.append(p_text)

        text = "\n".join(paragraphs)
        if not text:
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
        logger.error(f"Failed to extract document {path}: {e}")
        status = "UNSUPPORTED_FORMAT" if ext == ".doc" else "CORRUPT_FILE"
        return {
            "text": "",
            "status": status,
            "error": str(e)
        }