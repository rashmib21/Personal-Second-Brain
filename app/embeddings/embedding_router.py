from app.embeddings.text_embedding import embed_text
from app.embeddings.image_embedding import embed_image

def generate_embedding(file_type, content):
	#Generate embedding based on file_type

	#Text based
	if file_type in (
		"pdf","document","text","audio","video_transcript",):
		return embed_text(content)

	#Image based 
	elif file_type in ("image","video_frame",):
		return embed_image(content)

	raise ValueError(f"Unspported file type: {file_type}")	
