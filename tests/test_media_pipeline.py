import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.extractors.image import extract_image
from app.extractors.audio import extract_audio
from app.extractors.video import extract_video
from app.embeddings.embedding_router import generate_embedding

def test_media():
    print("=" * 70)
    print("MEDIA PIPELINE VERIFICATION (IMAGE, AUDIO, VIDEO)")
    print("=" * 70)

    # 1. Image OCR Test
    print("\n1. Testing Image Extractor & Embedding:")
    sample_img = "watched_folder/vingsfire.png"
    if os.path.exists(sample_img):
        res = extract_image(sample_img)
        print("  - OCR Status:", res.get("status"))
        print("  - Extracted Text Snippet:", repr(res.get("text", "")[:100]))
        if res.get("status") == "SUCCESS":
            vec = generate_embedding("image", res["text"])
            print("  - Generated Vector Dim:", len(vec))
            assert len(vec) == 384
    else:
        print(f"  - Sample image {sample_img} not found, testing exception fallback.")
        res = extract_image("non_existent.png")
        print("  - Fallback Status:", res.get("status"))
        assert res.get("status") == "CORRUPT_FILE"

    # 2. Audio ASR Test
    print("\n2. Testing Audio Extractor & Embedding:")
    sample_audio = "/home/rashmi/Downloads/audio.mpeg"
    if os.path.exists(sample_audio):
        res = extract_audio(sample_audio)
        print("  - ASR Status:", res.get("status"))
        print("  - Transcript Snippet:", repr(res.get("transcript", "")[:100]))
        if res.get("status") == "SUCCESS":
            vec = generate_embedding("audio", res["transcript"])
            print("  - Generated Vector Dim:", len(vec))
            assert len(vec) == 384
    else:
        print("  - Sample audio file not present, testing missing file fallback.")
        res = extract_audio("missing_audio.wav")
        print("  - Fallback Status:", res.get("status"))

    # 3. Video Processing Test
    print("\n3. Testing Video Extractor & Keyframe Cleanup:")
    sample_video = "/home/rashmi/Downloads/momo.mp4"
    if os.path.exists(sample_video):
        res = extract_video(sample_video)
        print("  - Video Status:", res.get("status"))
        print("  - Transcript Snippet:", repr(res.get("transcript", "")[:100]))
        print("  - Keyframes Extracted Count:", len(res.get("keyframes", [])))
        if res.get("status") == "SUCCESS":
            vec = generate_embedding("video", res["transcript"])
            print("  - Generated Vector Dim:", len(vec))
            assert len(vec) == 384
    else:
        print("  - Sample video file not present, testing fallback.")
        res = extract_video("missing_video.mp4")
        print("  - Fallback Status:", res.get("status"))

    print("\n" + "=" * 70)
    print("MEDIA PIPELINE VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_media()
