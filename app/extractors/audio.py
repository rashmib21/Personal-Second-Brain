import whisper

#Load the whisper model once when this module is imported
whisper_model=whisper.load_model("base")

def extract_audio(path):
	
	#Transcribe the audio
	result=whisper_model.transcribe(path, fp16=False,
    verbose=True)
	print(result["language"])
	print(result["text"])



	#Return only the text
	return result['text']

#Testing
if __name__=='__main__':
	file_path='watched_folder/MLKDream.mp3'
	text=extract_audio(file_path)
	print(text)		