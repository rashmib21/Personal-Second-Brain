import os
from app.extractors.pdf import extract_pdf
from app.extractors.text import extract_txt
from app.extractors.docx import extract_docx
def extract_text(path):
	#Detect file extension and call the correct extractor
	#Split filename and extension
	_, extension=os.path.splitext(path)

	#convert to lowercase
	extension=extension.lower()

	print(f"Detected extension: {extension}")

	if extension==".pdf":
		return extract_pdf(path)
	elif extension in [".txt", ".log", ".md", ".json", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf"]:
		return extract_txt(path)	
	elif extension in [".docx", ".doc"]:
		return extract_docx(path)
			
	raise ValueError(f"Unsupported file type: {extension}")	