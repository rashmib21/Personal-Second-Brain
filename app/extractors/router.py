import logging
import os

from app.extractors.pdf import extract_pdf
from app.extractors.text import extract_txt
from app.extractors.docx import extract_docx
from app.extractors.spreadsheet import extract_spreadsheet
from app.extractors.presentation import extract_presentation
from app.extractors.image import extract_image
from app.extractors.audio import extract_audio
from app.extractors.video import extract_video
from app.extractors.archive import extract_archive

logger = logging.getLogger(__name__)

def extract_file(path):
    _, extension = os.path.splitext(path)
    extension = extension.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    elif extension in [".txt", ".log", ".md", ".rtf"]:
        return extract_txt(path)

    elif extension in [".doc", ".docx", ".odt"]:
        return extract_docx(path)

    elif extension in [".csv", ".xlsx", ".xls", ".ods"]:
        return extract_spreadsheet(path)

    elif extension in [".ppt", ".pptx"]:
        return extract_presentation(path)

    elif extension in [
        ".jpg", ".jpeg", ".png",
        ".bmp", ".gif",
        ".tiff", ".tif",
        ".webp"
    ]:
        return extract_image(path)

    elif extension in [
        ".mp3", ".wav", ".m4a",
        ".aac", ".flac",
        ".ogg", ".opus"
    ]:
        return extract_audio(path)

    elif extension in [
        ".mp4", ".avi", ".mov",
        ".mkv", ".wmv",
        ".webm", ".mpeg", ".mpg", ".m4v"
    ]:
        return extract_video(path)

    elif extension in [
        ".zip", ".tar", ".gz", ".tgz"
    ]:
        return extract_archive(path)

    logger.warning(f"Unsupported file type: {extension} — skipping file: {path}")
    return {
        "text": "",
        "status": "UNSUPPORTED_FORMAT",
        "error": f"Unsupported extension {extension}"
    }