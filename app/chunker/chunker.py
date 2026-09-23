def split_text_recursively(text, max_chars, separators=None):
	"""
	Recursively splits text into small units using natural hierarchical separators:
	Paragraphs ("\n\n"), Lines ("\n"), Sentences (". "), Words (" ").
	"""
	if not text:
		return []

	cleaned = text.strip()
	if not cleaned:
		return []

	if len(cleaned) <= max_chars:
		return [cleaned]

	if separators is None:
		separators = ["\n\n", "\n", ". ", " "]

	if not separators:
		# Fallback: slice by character length if no separators left
		units = []
		for i in range(0, len(cleaned), max_chars):
			units.append(cleaned[i : i + max_chars])
		return units

	current_separator = separators[0]
	next_separators = separators[1:]

	raw_parts = cleaned.split(current_separator)
	units = []

	for part in raw_parts:
		part = part.strip()
		if not part:
			continue

		if len(part) <= max_chars:
			units.append(part)
		else:
			# Recursively split large part using finer separators
			sub_units = split_text_recursively(part, max_chars, next_separators)
			for sub_unit in sub_units:
				units.append(sub_unit)

	return units


def get_tail_overlap(text_unit, needed_chars):
	"""
	Extracts a clean tail snippet of length up to needed_chars without cutting words in half.
	"""
	if not text_unit or needed_chars <= 0:
		return ""

	if len(text_unit) <= needed_chars:
		return text_unit

	raw_tail = text_unit[-needed_chars:]
	# Find the first space character so we do not start mid-word
	first_space_index = raw_tail.find(" ")
	if first_space_index != -1 and first_space_index < len(raw_tail) - 1:
		clean_tail = raw_tail[first_space_index + 1 :]
		if clean_tail:
			return clean_tail

	return raw_tail


def chunk_text(text, max_chars=600, overlap_chars=120):
	"""
	Splits text using Recursive Structural Chunking with Sliding Overlap.

	Parameters:
		text (str): Input text from document or image OCR/VLM.
		max_chars (int): Maximum character length per chunk (default: 600).
		overlap_chars (int): Character overlap window between adjacent chunks (default: 120).

	Returns:
		list of str: List of overlapping text chunks.
	"""
	# Step 1: Handle empty or whitespace-only text
	if not text:
		return []

	cleaned_text = text.strip()
	if not cleaned_text:
		return []

	# If entire text fits inside max_chars, return it as a single chunk
	if len(cleaned_text) <= max_chars:
		return [cleaned_text]

	# Step 2: Recursively break text into natural sub-units
	sub_units = split_text_recursively(cleaned_text, max_chars)

	if not sub_units:
		return []

	# Step 3: Combine sub-units into chunks with sliding overlap
	chunks = []
	current_units = []
	current_length = 0

	for unit in sub_units:
		unit_length = len(unit)

		# Check if adding this unit exceeds max_chars
		separator_needed = 1 if current_units else 0
		if current_length + unit_length + separator_needed <= max_chars:
			current_units.append(unit)
			current_length += unit_length + separator_needed
		else:
			# Complete the current chunk
			if current_units:
				completed_chunk = "\n".join(current_units)
				chunks.append(completed_chunk)

			# Build sliding overlap from previous units
			overlap_units = []
			accumulated_overlap = 0

			# Work backwards to gather units/snippets for the overlap window
			for prev_unit in reversed(current_units):
				needed_chars = overlap_chars - accumulated_overlap
				if needed_chars <= 0:
					break

				if len(prev_unit) <= needed_chars:
					overlap_units.insert(0, prev_unit)
					accumulated_overlap += len(prev_unit) + 1
				else:
					snippet = get_tail_overlap(prev_unit, needed_chars)
					if snippet:
						overlap_units.insert(0, snippet)
						accumulated_overlap += len(snippet) + 1
					break

			# Start the new chunk with overlap units plus the current unit
			current_units = overlap_units + [unit]
			current_length = sum(len(u) for u in current_units) + (len(current_units) - 1)

	# Step 4: Add remaining units as the final chunk
	if current_units:
		final_chunk = "\n".join(current_units)
		if not chunks or chunks[-1] != final_chunk:
			chunks.append(final_chunk)

	return chunks