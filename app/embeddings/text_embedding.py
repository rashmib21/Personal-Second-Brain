#Responsible for PDF, DOCX, TXT, OCR Text, Audio Transcript, Video Transcript
from sentence_transformers import SentenceTransformer

#Load the model only once
model=SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
	#Convert text into a vector embeddings
	embedding=model.encode(
		text,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)

	return embedding.tolist()

#Testing
if __name__=="__main__":
	sample_text="""
	Artificial Intelligence has transformed the way humans interact with computers by enabling machines to understand language, recognize images, and make intelligent decisions. Modern AI systems rely on large amounts of data to learn meaningful patterns instead of following manually written rules. One of the most important concepts in natural language processing is the embedding, which converts words, sentences, or even entire documents into numerical vectors that preserve semantic meaning. These vectors allow computers to compare the similarity between different pieces of text using mathematical operations. For example, two sentences discussing machine learning may have very different wording but still produce embeddings that are close together in vector space because they express similar ideas. This capability is essential for applications such as semantic search, recommendation systems, chatbots, and retrieval-augmented generation. Transformer-based models create these embeddings by processing tokens through multiple self-attention layers, allowing each word to understand the context provided by surrounding words. The final embedding is a compact numerical representation of the entire text, regardless of whether the input contains a few words or several paragraphs. These embeddings are then stored in specialized vector databases, where similarity search algorithms efficiently retrieve the most relevant information. As embedding models continue to improve, AI systems become better at understanding context, intent, and relationships between concepts rather than relying solely on exact keyword matching. This advancement has significantly improved the quality of modern search engines, virtual assistants, and intelligent document retrieval systems. Consequently, embeddings have become one of the foundational building blocks of modern generative AI applications."""
	embedding=embed_text(sample_text)
	print(f"Embedding Dimension: {len(embedding)}")
	print(embedding)
	print("****************")
	print(len(set(embedding)))