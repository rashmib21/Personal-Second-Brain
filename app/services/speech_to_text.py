from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(audio_path):

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )

        transcript = ""

        for segment in segments:
            transcript += segment.text + " "

        return {
            "text": transcript.strip(),
            "language": info.language,
            "words": []
        }

    except Exception as e:
        raise RuntimeError(f"Speech-to-text failed: {e}")











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