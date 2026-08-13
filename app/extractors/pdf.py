import pdfplumber

def extract_pdf(path):
	"""
	Extracts text from a PDF file page by page.
	Returns a list of page dictionaries with page numbers.
	"""
	pages_data = []

	try:
		with pdfplumber.open(path) as pdf:
			for i, page in enumerate(pdf.pages, start=1):
				page_text = page.extract_text() or ""
				page_text = page_text.strip()

				if page_text:
					pages_data.append({
						"page": i,
						"text": page_text
					})
	except Exception as e:
		print(f"PDF extraction error: {e}")

	return pages_data
