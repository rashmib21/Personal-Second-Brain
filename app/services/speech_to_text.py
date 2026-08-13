import torch
from qwen_asr import Qwen3ASRModel

model_name="moorlee/qwen3-asr-0.6b-hinglish"

model = None

def get_asr_model():
	global model
	if model is None:
		print("Loading Srota ASR model...")
		model = Qwen3ASRModel.from_pretrained(
			model_name,
			dtype=torch.float16,
			device_map="cuda:0"
		)
		print("Srota ASR model loaded.")
	return model


def transcribe_audio(audio_path):

    try:
        asr_model = get_asr_model()
        result = asr_model.transcribe(audio_path)

        if not result:
            return {
                "text":"",
                "language":"",
                "words":[]
            }

        transcription=result[0]

        return {
            "text": transcription.text,
            "language": transcription.language or "unknown",
            "words": []
        }

    except Exception as e:
        raise RuntimeError(f"Speech-to-text failed: {e}")




# Testing
# if __name__ == "__main__":

#     file_path = "/home/rashmi/Downloads/audio.mpeg"

#     print("=" * 70)
#     print("Testing Srota ASR")
#     print("=" * 70)

#     result = transcribe_audio(file_path)

#     print("\nDetected Language:", result["language"])

#     print("\nTranscript:")
#     print(result["text"])

#     print("\nWords:")
#     print(result["words"])

    print("=" * 70)






# import os
# from elevenlabs.client import ElevenLabs
# from config import ELEVENLABS_API_KEY

# #Create the client once when the module is imported
# client=ElevenLabs(
# 	api_key=ELEVENLABS_API_KEY
# 	)

# def transcribe_audio(audio_path):
# 	#Transcribe an audio file using elevenlabs scribe
# 	try:
# 		with open(audio_path,'rb') as audio_file:
# 			response=client.speech_to_text.convert(
# 				file=audio_file,
# 				model_id="scribe_v2",
# 				diarize=True,
# 				tag_audio_events=True,
# 				#None =auto language detection
# 				language_code=None
# 				)
# 		return {
# 			"text":response.text,
# 			"language":response.language_code,
# 			"words":response.words,
# 		}	
# 	except Exception as e:
# 		raise RuntimeError(f"Speech-to-text failed: {e}")	


#Testing
# if __name__ == "__main__":

#     file_path = "watched_folder/MLKDream.mp3"

#     result = transcribe_audio(file_path)

#     print("=" * 70)
#     print("Language :", result["language"])
#     print("=" * 70)
#     print(result["text"])