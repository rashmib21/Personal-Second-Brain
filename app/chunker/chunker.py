def chunk_text(text, chunk_size=500):

	#Return empty list if there is no text
	if not text:
		return []

	#split text into chunks of fixed size
	chunks=[]

	for i in range(0, len(text), chunk_size):
		chunk=text[i:i + chunk_size]
		chunks.append(chunk.strip())
	return chunks	