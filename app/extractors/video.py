# Responsible for extracting transcript and keyframes from video files

import os
import subprocess
import cv2

from app.extractors.audio import extract_audio


def extract_keyframes(path, interval_sec=8):

    # Open the video
    video = cv2.VideoCapture(path)

    # Frames per second
    fps = video.get(cv2.CAP_PROP_FPS)

    # Number of frames to skip
    frame_gap = int(fps * interval_sec)

    frame_number = 0

    # Store keyframe paths
    keyframes = []

    while video.isOpened():

        success, frame = video.read()

        if not success:
            break

        # Save one frame every interval_sec
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

    # Temporary audio file
    audio_path = os.path.splitext(path)[0] + ".wav"

    # Extract audio using FFmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            path,
            "-y",
            audio_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Convert speech to text
    transcript = extract_audio(audio_path)

    # Delete temporary audio
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Extract keyframes
    keyframes = extract_keyframes(path)

    # Return extracted data
    return {
        "transcript": transcript,
        "keyframes": keyframes,
    }


# # Testing
# if __name__ == "__main__":
#
#     file_path = "watched_folder/sample.mp4"
#
#     result = extract_video(file_path)
#
#     print("\n----- Transcript -----\n")
#     print(result["transcript"])
#
#     print("\n----- Keyframes -----\n")
#
#     for frame in result["keyframes"]:
#         print(frame)