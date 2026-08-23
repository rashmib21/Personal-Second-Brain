import warnings
from PIL import Image
from transformers import logging as tf_logging
from sentence_transformers import SentenceTransformer

# Suppress Hugging Face transformers warnings regarding slow image processor defaults
tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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
if __name__=="__main__":
	image_path="watched_folder/ChatGPT Image Jul 29, 2026, 03_10_26 PM.png"
	embedding=embed_image(image_path)
	print(f"Embedding Dimension: {len(embedding)}")
	print(embedding)
