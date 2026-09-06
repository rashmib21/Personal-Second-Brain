import os
import subprocess
import tempfile
import torch
from qwen_asr import Qwen3ASRModel

model_name = "moorlee/qwen3-asr-0.6b-hinglish"

model = None

def get_asr_model():
	global model
	if model is None:
		print("Loading Srota ASR model...")
		model = Qwen3ASRModel.from_pretrained(
			model_name,
			dtype=torch.bfloat16,
			device_map="cuda:0",
			max_inference_batch_size=1
		)
		print("Srota ASR model loaded.")
	return model


def transcribe_audio(audio_path):

    try:
        asr_model = get_asr_model()

        # Check audio duration using ffprobe if available
        duration = None
        if os.path.exists(audio_path):
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    audio_path
                ]
                res_dur = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                duration = float(res_dur.stdout.strip())
            except Exception:
                duration = None

        # For long audio (> 60s), split into 60s segments to prevent GPU VRAM OOM on low-VRAM setup
        if duration and duration > 60.0:
            segment_length = 60.0
            full_text_parts = []
            detected_lang = "unknown"

            with tempfile.TemporaryDirectory() as tmpdir:
                start_sec = 0.0
                seg_idx = 0
                while start_sec < duration:
                    seg_file = os.path.join(tmpdir, f"seg_{seg_idx}.wav")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_sec),
                        "-t", str(segment_length),
                        "-i", audio_path,
                        "-ar", "16000",
                        "-ac", "1",
                        "-c:a", "pcm_s16le",
                        seg_file
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                    res = asr_model.transcribe(audio=seg_file)
                    if res:
                        t_obj = res[0]
                        if t_obj.text:
                            full_text_parts.append(t_obj.text.strip())
                        if t_obj.language and detected_lang == "unknown":
                            detected_lang = t_obj.language
                    start_sec += segment_length
                    seg_idx += 1
                    torch.cuda.empty_cache()

            full_text = "\n".join(full_text_parts)
            return {
                "text": full_text,
                "language": detected_lang,
                "words": []
            }

        # For short audio (<= 60s), transcribe directly
        result = asr_model.transcribe(audio=audio_path)

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