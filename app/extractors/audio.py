import logging
from app.services.speech_to_text import transcribe_audio

logger = logging.getLogger(__name__)

def extract_audio(path):
    """
    Transcribe audio file using speech-to-text service.
    Returns structured dict with transcript and status.
    """
    try:
        result = transcribe_audio(path)
        transcript = (result.get("text") or "").strip()
        language = result.get("language") or "unknown"

        if not transcript:
            return {
                "status": "NO_TEXT_EXTRACTED",
                "transcript": "",
                "text": "",
                "language": language,
                "error": None
            }

        return {
            "status": "SUCCESS",
            "transcript": transcript,
            "text": transcript,
            "language": language,
            "error": None
        }
    except RuntimeError as e:
        logger.error(f"Audio transcription failed for {path}: {e}")
        return {
            "status": "MISSING_DEPENDENCY" if "model" in str(e).lower() else "PROCESSING_ERROR",
            "transcript": "",
            "text": "",
            "language": "unknown",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Failed to extract audio {path}: {e}")
        return {
            "status": "CORRUPT_FILE",
            "transcript": "",
            "text": "",
            "language": "unknown",
            "error": str(e)
        }