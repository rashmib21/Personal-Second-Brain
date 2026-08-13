import logging
import pdfplumber

logger = logging.getLogger(__name__)

def extract_pdf(path):
    """
    Extracts text from a PDF file page by page.
    Returns a structured dictionary with page list and status.
    """
    pages_data = []
    error_msg = None

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                if page_text:
                    pages_data.append({
                        "page": i,
                        "text": page_text
                    })
    except Exception as e:
        logger.error(f"PDF extraction error for {path}: {e}")
        error_msg = str(e)
        return {
            "status": "CORRUPT_FILE",
            "pages": [],
            "text": "",
            "error": error_msg
        }

    if not pages_data:
        return {
            "status": "NO_TEXT_EXTRACTED",
            "pages": [],
            "text": "",
            "error": "No extractable text found in PDF pages."
        }

    full_text = "\n".join(p["text"] for p in pages_data)

    return {
        "status": "SUCCESS",
        "pages": pages_data,
        "text": full_text,
        "error": None
    }

