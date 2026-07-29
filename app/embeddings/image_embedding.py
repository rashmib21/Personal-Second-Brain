from PIL import Image
from sentence_transformers import SentenceTransformer

#Load clip model
clip_model = SentenceTransformer("clip-ViT-B-32")

def embed_image(image_path):
	#Generate an ebmbedding for an image
	image=Image.open(image_path).convert("RGB")
	embedding=clip_model.encode(image)
	return embedding.tolist()

#Testing
if __name__=="__main__":
	image_path="watched_folder/passport_size.png"
	embedding=embed_image(image_path)
	print(f"Embedding Dimension: {len(embedding)}")
	print(embedding)
