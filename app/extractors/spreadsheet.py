def extract_spreadsheet(path):
	with open(path, 'r', encoding='utf-8'):
		spreadsheet=file.read()
	return spreadsheet	