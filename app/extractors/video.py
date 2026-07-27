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

	#Number of frames to skip
	frame_gap=int(fps*interval_sec)

	frame_number=0
	keyframes=[]


	while video.isOpened():

		success, frame=video.read()

		if not success:
			break

		if frame_number	% frame_gap==0:
			frame_name=(os.path.splitext(path)[0]+f"_frame_{frame_name}.jpg")

			cv2.imwrite(frame_name,frame)

			keyframes.append(frame_name)
		frame_number=frame_number+1
		
	video.release()

	return keyframes		