import torch
from qwen_asr import Qwen3ASRModel

audio_file = "/home/rashmi/Downloads/audio.mpeg"

print("Loading model...")

model = Qwen3ASRModel.from_pretrained(
    "moorlee/qwen3-asr-0.6b-hinglish",
    dtype=torch.float16,
    device_map="cpu",
)

print("Model loaded!")
print("Transcribing...")

result = model.transcribe(audio_file)

print("\n===== RAW RESULT =====")
print(repr(result))
print("======================")
