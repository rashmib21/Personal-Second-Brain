def extract_text(path):

	#extract text from a plain text .txt file
	#open the file
	with open(path,'r', encoding='utf-8') as file:
		#Read the complete file
		text=file.read()
	return text	