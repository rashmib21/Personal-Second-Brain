from app.embeddings.text_embedding import embed_text

def generate_embedding(file_type, content):
    """
    Generates a 384-dimensional text embedding for all text chunks
    (PDF, DOCX, TXT, OCR text, ASR transcripts, Archives, Presentations).
    """
    if not isinstance(content, str):
        content = str(content)

    return embed_text(content)

