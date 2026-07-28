#Responsible for extracting text from images

from PIL import Image #Open image files
import pytesseract #OCR Engine


def extract_image(path):
	
	#open the image
	image=Image.open(path)

	#OCR
	text=pytesseract.image_to_string(image)

	#return the text
	return text	

#Testing 

if __name__=="__main__":
	file_path="watched_folder/vingsfire.png"
	text=extract_image(file_path)
	print(text)	