import pdfplumber #copying text from pdf
import pytesseract #scan a paper and make pdf, pytesseract performs OCR 
from pdf2image import convert_from_path #OCR can't work directly on a pdf

# from app.chunker.chunker import chunk_text

def extract_pdf(path):
	#Read a pdf file and return all its text, if the pdf contains a real text layer, use pdfplumber
	#Otherwise OCR

	#Store extracted text
	text=""

	#Open the pdf
	with pdfplumber.open(path) as pdf:
		#Visit every page
		for page in pdf.pages:
			#Read text from the page
			page_text=page.extract_text() or ""

			#Add it to the final result
			text=text+ page_text + "\n"
	#If we successfully extracted text, no need for OCR
	if text.strip():
		return text

	#Convert scanned pdf into images
	images=convert_from_path(path)		

	#Store OCR output
	ocr_text=""

	#Read every page image
	for image in images:

		#Extract text using OCR
		page_text=pytesseract.image_to_string(image)

		ocr_text=ocr_text+page_text+"\n"
	return ocr_text

# if __name__=="__main__":
# 	file_path="watched_folder/Rashmi_Barethiya_19-05.pdf"

# 	text=extract_text(file_path)
# 	# print(text)
# 	chunks=chunk_text(text)
# 	print(f"Total chunks: {len(chunks)}")

# 	for i, chunk in enumerate(chunks, start=1):
# 		print(f"\n\n\n\n\n-----Chunks {i}-----")
# 		print(chunk)

