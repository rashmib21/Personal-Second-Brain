import pdfplumber #copying text from pdf
import pytesseract #scan a paper and make pdf, pytesseract performs OCR 
from pdf2image import convert_from_path #OCR can't work directly on a pdf


def extract_text(path):
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