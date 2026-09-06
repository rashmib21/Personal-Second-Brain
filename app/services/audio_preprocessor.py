import os
import shutil
import subprocess
import tempfile


def preprocess_audio(audio_path, output_path=None):
    """
    Preprocesses an input audio file into a standardized WAV file.
    
    Conversion specifications:
    - Sample Rate: 16 kHz (16000 Hz)
    - Channels: 1 (Mono)
    - Format: WAV (PCM 16-bit)
    """
    # 1. Check whether the original input file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 2. Check whether ffmpeg and ffprobe are available on the system
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not available in system PATH.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is not installed or not available in system PATH.")

    # 3. Handle output path: if not specified, create a temporary WAV file
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_file.name
        temp_file.close()

    # 4. Prepare ffmpeg command to convert audio
    # -y                     : Overwrite output file without asking
    # -i audio_path          : Input file path
    # -ar 16000              : Resample audio to 16000 Hz (16 kHz)
    # -ac 1                  : Set audio channels to 1 (mono)
    # -c:a pcm_s16le         : Use PCM 16-bit little-endian codec
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.strip()}") from e

    # 5. Use ffprobe to get the duration of the processed audio
    # -v error                              : Suppress all output except errors
    # -show_entries format=duration         : Extract only the duration field
    # -of default=noprint_wrappers=1:nokey=1: Print numeric duration value only
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        output_path
    ]

    try:
        result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration using ffprobe: {e}") from e

    # 6. Return standard metadata dictionary
    return {
        "original_path": audio_path,
        "processed_path": output_path,
        "duration": duration,
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "sample_format": "pcm_s16le"
    }
