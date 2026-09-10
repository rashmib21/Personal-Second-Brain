import os
import re
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
                            cleaned_part = clean_asr_hallucination_loops(t_obj.text.strip())
                            if cleaned_part:
                                full_text_parts.append(cleaned_part)
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
        cleaned_text = clean_asr_hallucination_loops(transcription.text)

        return {
            "text": cleaned_text,
            "language": transcription.language or "unknown",
            "words": []
        }

    except Exception as e:
        raise RuntimeError(f"Speech-to-text failed: {e}")


def clean_asr_hallucination_loops(raw_text):
    """
    Cleans transcript text by stripping ASR hallucination loops, 
    repeating phrases, repeating Hindi/English patterns, and duplicate line fragments.
    Follows beginner-friendly guidelines with explicit logic steps and comments.
    """
    if not raw_text:
        return ""

    # Step 1: Remove known ASR hallucination pattern strings (Hindi and English)
    text = raw_text
    text = re.sub(r"(?:आप\s*देखने\s*के\s*लिए\s*धन्यवाद\s*)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:देखने\s*के\s*लिए\s*धन्यवाद\s*)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:धन्यवाद\s*){3,}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?:और|इसके\s*बाद)\s*)?यहाँ\s*आपको\s*एक\s*बार\s*फिर\s*इसके\s*बारे\s*में\s*सीखना\s*है\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?:अब\s*)?आप\s*देख\s*सकते\s*हैं\s*कि\s*यह\s*एक\s*बड़ा\s*विवरण\s*बना\s*रहा\s*है\s*(?:जो)?\s*)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?:अतः\s*यह\s*एक\s*नियतकार्य\s*है\s*जो\s*)?आप\s*देख\s*सकते\s*हैं\s*जैसे\s*कि\s*)?(?:और\s*)?आप\s*इस\s*पेज\s*पर\s*जाते\s*हैं\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:आर्गेनिक\s*रिडाइवेक्टिंग\s*)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:अधिकतम\s*वर्गीकृत\s*\d*%?\s*)+", "", text, flags=re.IGNORECASE)

    # Step 2: Remove repeating single words (3 or more consecutive identical words, e.g. "so so so so", "right right right")
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)

    # Step 3: Remove repeating short & long phrase loops (1 to 15 words repeating 1 or more times)
    for _ in range(3):
        text = re.sub(r"\b((?:\w+\s+){1,15}\w+)(?:\s+\1){1,}\b", r"\1", text, flags=re.IGNORECASE)

    # Step 4: Line by line cleaning & deduplication
    raw_lines = text.split("\n")
    processed_lines = []
    previous_line_lower = None

    for line_item in raw_lines:
        stripped_line = line_item.strip()

        # Skip empty lines
        if not stripped_line:
            continue

        # Skip if identical to previous line
        current_line_lower = stripped_line.lower()
        if current_line_lower == previous_line_lower:
            continue

        # Check for phrase repetitions within line again after sentence splitting
        words_in_line = stripped_line.split()
        if len(words_in_line) >= 4:
            # Deduplicate contiguous repeated phrase chunks inside single line
            unique_line_words = []
            word_idx = 0
            while word_idx < len(words_in_line):
                matched_phrase_len = 0
                max_check_len = min(10, (len(words_in_line) - word_idx) // 2)
                for phrase_len in range(max_check_len, 0, -1):
                    first_slice = words_in_line[word_idx : word_idx + phrase_len]
                    second_slice = words_in_line[word_idx + phrase_len : word_idx + 2 * phrase_len]
                    if [w.lower() for w in first_slice] == [w.lower() for w in second_slice]:
                        matched_phrase_len = phrase_len
                        break

                if matched_phrase_len > 0:
                    unique_line_words.extend(words_in_line[word_idx : word_idx + matched_phrase_len])
                    word_idx += 2 * matched_phrase_len
                else:
                    unique_line_words.append(words_in_line[word_idx])
                    word_idx += 1

            stripped_line = " ".join(unique_line_words)

        if stripped_line:
            processed_lines.append(stripped_line)
            previous_line_lower = current_line_lower

    final_cleaned_transcript = "\n".join(processed_lines).strip()
    return final_cleaned_transcript





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