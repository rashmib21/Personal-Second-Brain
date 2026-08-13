import logging
import os
import subprocess
import cv2
from app.extractors.audio import extract_audio

logger = logging.getLogger(__name__)

def extract_keyframes(path, interval_sec=8):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 1

    frame_gap = max(1, int(fps * interval_sec))
    frame_number = 0
    keyframes = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if frame_number % frame_gap == 0:
            frame_name = (
                os.path.splitext(path)[0]
                + f"_frame_{frame_number}.jpg"
            )
            cv2.imwrite(frame_name, frame)
            keyframes.append(frame_name)

        frame_number += 1

    video.release()
    return keyframes


def extract_video(path):
    """
    Extracts transcript from video files via FFmpeg and ASR.
    Returns structured dict with status.
    """
    audio_path = os.path.splitext(path)[0] + "_temp_audio.wav"
    transcript = ""
    keyframes = []

    try:
        # Step 1: Extract audio using FFmpeg with try/finally cleanup
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    audio_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                audio_result = extract_audio(audio_path)
                transcript = audio_result.get("transcript") or audio_result.get("text", "")
            else:
                logger.info(f"No audio stream found in video: {path}")

        except FileNotFoundError:
            logger.error("FFmpeg executable not found on system path.")
            return {
                "status": "MISSING_DEPENDENCY",
                "transcript": "",
                "text": "",
                "keyframes": [],
                "error": "FFmpeg not installed"
            }
        except subprocess.CalledProcessError:
            logger.info(f"FFmpeg failed to extract audio from video (likely silent video): {path}")

        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp audio file {audio_path}: {e}")

        # Step 2: Extract keyframes safely
        try:
            keyframes = extract_keyframes(path)
        except Exception as e:
            logger.warning(f"Keyframe extraction error for {path}: {e}")

        transcript = transcript.strip()
        if not transcript:
            return {
                "status": "NO_TEXT_EXTRACTED",
                "transcript": "",
                "text": "",
                "keyframes": keyframes,
                "error": "No speech transcript found in video"
            }

        return {
            "status": "SUCCESS",
            "transcript": transcript,
            "text": transcript,
            "keyframes": keyframes,
            "error": None
        }

    except Exception as e:
        logger.error(f"Failed to process video {path}: {e}")
        return {
            "status": "CORRUPT_FILE",
            "transcript": "",
            "text": "",
            "keyframes": [],
            "error": str(e)
        }