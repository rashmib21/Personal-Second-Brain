from  app.services.speech_to_text import transcribe_audio

def extract_audio(path):

	# Transcribe audio
    result = transcribe_audio(path)



    # Print detected language
    print(f"Detected language: {result['language']}")

    # Return transcript
    return {
        "transcript": result["text"],
        "language": result["language"],
        "words": result["words"]
    }

		

#Testing
# if __name__=='__main__':
# 	file_path='/home/rashmi/Downloads/audio.mpeg'
# 	text=extract_audio(file_path)
# 	print(text)		