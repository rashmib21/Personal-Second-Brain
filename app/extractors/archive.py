def extract_archive(path):
	with open(path, 'r', encoding='utf-8'):
		archive=file.read()
	return archive	