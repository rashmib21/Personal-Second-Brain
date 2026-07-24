import os
from app.extractors.pdf import extract_pdf
from app.extractors.text import extract_txt
from app.extractors.docx import extract_docx
from app.extractors.spreadsheet import extract_spreadsheet
from app.extractors.image import extract_image
from app.extractors.audio import extract_audio
from app.extractors.video import extract_video
from app.extractors.archive import extract_archive

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
	elif extension in [".csv", ".xlsx", ".xls", ".ods"]:
		return extract_spreadsheet(path)
	elif extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".heic"]:
		return extract_image(path)
	elif extension in [".mp3", ".wav", ".m4a",".aac", ".flac", ".ogg", ".wma"]:
		return extract_audio(path)
	elif extension in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".3gp"]	
		return extract_video(path)
	elif extension in [".zip", ".rar", ".7z",".tar", ".gz"]:
		return extract_archive(path)	

	

	raise ValueError(f"Unsupported file type: {extension}")	