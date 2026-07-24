def extract_image(path):
	with open(path, 'r', encoding='utf-8'):
		image=file.read()
	return image	