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
			frame_name=(os.path.splitext(path)[0]+f"_frame_{frame_number}.jpg")

			cv2.imwrite(frame_name,frame)

			keyframes.append(frame_name)
		frame_number=frame_number+1

	video.release()

	return keyframes		

def extract_video(path):
	#Extract audio
	audio_path=os.path.splitext(path)[0]+".wav"

	subprocess.run(
		[
			"ffmpeg",
			"-i",
			path,
			"-y",
			audio_path,
		],
		check=True,
	)

	#Speech to Text
	transcript=extract_audio(audio_path)

	#Delete temporary audio
	os.remove(audio_path)

	#Extract keyframes
	keyframes=extract_keyframes(path)

	return {
		"transcript":transcript,
		"keyframes":keyframes,
	}	

#Testing
if __name__=="__main__":
	file_path="watched_folder/sample1.mp4"
	result=extract_video(file_path)

	print("\n------Transcript------\n")
	print(result["transcript"])

	print("\n------Keyframes------\n")
	
	for frame in result['keyframes']:
		print(frame)
