# import whisper
from faster_whisper import WhisperModel

#Load the whisper model once when this module is imported
model=WhisperModel(
	"medium",
	device="cpu", 
	compute_type="int8")

def extract_audio(path):

	segments, info=model.transcribe(path)
	print(f"Detected language: {info.language}")

	text=''
	for segment in segments:
		text=text+segment.text+" "
	return text.strip()
		

#Testing
if __name__=='__main__':
	file_path='watched_folder/MLKDream.mp3'
	text=extract_audio(file_path)
	print(text)		