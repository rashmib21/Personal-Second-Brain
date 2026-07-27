import subprocess #-run ffmpeg
import whisper  #convert speech to text


whisper_model=whisper.load_model('base')

def extract_video(path):
	with open(path, 'r', encoding='utf-8'):
		video=file.read()
	return video	