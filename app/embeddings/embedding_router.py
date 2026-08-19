from app.embeddings.text_embedding import embed_text
from app.embeddings.image_embedding import embed_image

def generate_embedding(file_type, content):
    """
    Generates a 384-dimensional text embedding for all text chunks
    (PDF, DOCX, TXT, OCR text, ASR transcripts, Archives, Presentations).
    """
    #Image embeddings
    if file_type=="image":
        return embed_image(content)

    #Text embeddings for all other file types
    if not isinstance(content, str):
        content = str(content)

    return embed_text(content)

