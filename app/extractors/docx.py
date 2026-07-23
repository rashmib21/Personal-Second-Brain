def extract_docx(path):
	with open(path, 'r', encoding='utf-8'):
		docx=file.read()
	return docx	