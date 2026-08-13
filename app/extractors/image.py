import logging
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

def extract_image(path):
    """
    Extracts text from image files using Tesseract OCR.
    Returns structured dict with status.
    """
    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image) or ""
        text = text.strip()

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

    except pytesseract.TesseractNotFoundError as e:
        logger.error(f"Tesseract OCR executable not found: {e}")
        return {
            "text": "",
            "status": "MISSING_DEPENDENCY",
            "error": "Tesseract OCR binary not installed"
        }
    except Exception as e:
        logger.error(f"Failed to process image {path}: {e}")
        return {
            "text": "",
            "status": "CORRUPT_FILE",
            "error": str(e)
        }