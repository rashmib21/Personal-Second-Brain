import os
from app.extractors.pdf import extract_pdf

def extract_text(path):
	#Detect file extension and call the correct extractor
	#Split filename and extension
	_, extension=os.path.splitext(path)

	#convert to lowercase
	extension=extension.lower()

	print(f"Detected extension: {extension}")

	if extension==".pdf":
		return extract_pdf(path)
	raise ValueError(f"Unsupported file type: {extension}")	