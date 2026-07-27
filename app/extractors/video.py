import subprocess #-run ffmpeg
# import whisper  #convert speech to text
import os
import cv2

from app.extractors.audio import extract_audio

# whisper_model=whisper.load_model('base')

def extract_keyframes(path, interval_sec=8):
	
	#Open the video
	video=cv2.VideoCapture(path)

	#Frames per second
	fps=video.get(cv2.CAP_PROP_FPS)
	
