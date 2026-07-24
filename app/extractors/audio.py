def extract_audio(path):
	with open(path, 'r', encoding='utf-8'):
		audio=file.read()
	return audio	