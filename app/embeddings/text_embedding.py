#Responsible for PDF, DOCX, TXT, OCR Text, Audio Transcript, Video Transcript
from sentence_transformers import SentenceTransformer

#Load the model only once
model=SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def embed_text(text):
	#Convert text into a vector embeddings
	embedding=model.encode(
		text,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)

	return embedding.tolist()

# #Testing
# if __name__=="__main__":
# 	sample_text="watched_folder/hello.txt"
# 	embedding=embed_text(sample_text)
# 	print(f"Embedding Dimension: {len(embedding)}")
# 	print(embedding)
# 	print("*"*60)
# 	print(len(set(embedding)))