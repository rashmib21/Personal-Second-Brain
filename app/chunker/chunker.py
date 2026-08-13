def chunk_text(text, max_chars=600):
	"""
	Splits text into readable chunks by paragraphs so words are not cut in half.
	"""
	if not text:
		return []

	# Split text by line breaks
	paragraphs = text.split("\n")
	chunks = []
	current_chunk = ""

	for paragraph in paragraphs:
		paragraph = paragraph.strip()
		if not paragraph:
			continue

		# If adding this paragraph fits inside max_chars, append it
		if len(current_chunk) + len(paragraph) + 1 <= max_chars:
			if current_chunk:
				current_chunk += "\n" + paragraph
			else:
				current_chunk = paragraph
		else:
			# Save completed chunk and start a new one
			if current_chunk:
				chunks.append(current_chunk)
			current_chunk = paragraph

	# Add any remaining text
	if current_chunk:
		chunks.append(current_chunk)

	return chunks