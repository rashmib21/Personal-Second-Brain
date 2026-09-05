import logging 
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger=logging.getLogger(__name__)

TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1
TARGET_SAMPLE_FORMAT='pcm_s16le'

def check_ffmpeg():
	"Verify that ffmpeg is installed and available in PATH"
	if shutil.which('ffmpeg') is None:
		raise RuntimeError("ffmpeg is not installed or not available in PATH.")

def get_audio_duration(audio_path):
	"Return audio duration in seconds using ffprobe. Returns float duration in seconds"
	if not os.path.exists(audio_path):
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	check check_ffmpeg()
	command=[
		"ffprobe",
		"-v",
		"error",
		"-show_entries",
		"format=duration",
		"-of",
		"default=noprint_wrapper=1:nokey=1",
		audio_path
	]

	try:
		result=subprocess.run(command,capture_output=True, text=True, check=True)
		return float(result.stdout.strip())
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Unable to determine audio duration: {e.stderr.strip()}") from e
	except ValueError as e:
		raise RuntimeError(f"Invalid duration returned by ffprobe for: {audio_path}") from e

			