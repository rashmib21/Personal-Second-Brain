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

