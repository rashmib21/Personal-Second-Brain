import os
from app.embeddings.text_embedding import embed_text
from app.embeddings.image_embedding import embed_image

def generate_embedding(file_type, content):
    """
    Generates a 384-dimensional text embedding for all text chunks
    (PDF, DOCX, TXT, OCR text, ASR transcripts, Archives, Presentations, VLM summaries).
    """
    if file_type == "image" and isinstance(content, str) and len(content) < 4096 and os.path.exists(content):
        return embed_image(content)

    # Text embeddings for all text chunks
    if not isinstance(content, str):
        content = str(content)

    return embed_text(content)


