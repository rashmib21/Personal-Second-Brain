from PIL import Image
from sentence_transformers import SentenceTransformer

_clip_model = None

def get_clip_model():
	global _clip_model
	if _clip_model is None:
		_clip_model = SentenceTransformer("clip-ViT-B-32")
	return _clip_model

def embed_image(image_path):
	#Generate an embedding for an image
	model = get_clip_model()
	image=Image.open(image_path).convert("RGB")
	embedding=model.encode(image)
	return embedding.tolist()

# #Testing
# if __name__=="__main__":
# 	image_path="watched_folder/passport_size.png"
# 	embedding=embed_image(image_path)
# 	print(f"Embedding Dimension: {len(embedding)}")
# 	print(embedding)
