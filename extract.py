from PIL import Image
import numpy as np
import shutil
import os
import subprocess
from datetime import datetime
import whisper
from speaker import decide_speaker
from emotions import detect_emotions

video_path = 'TV Clips/nine-nine/nine-nine.mp4'

path_parts = video_path.split('/')
annotated_path = '/'.join(path_parts[:-1]) + '/annotated_images'

if os.path.exists(annotated_path):
    shutil.rmtree(annotated_path)
os.makedirs(annotated_path, exist_ok=True)

model = whisper.load_model("base")
transcript = model.transcribe(video_path)

# scores and emotions mapping
emotions_dict = {
    "Neutral": 0.5,
    "Happy": 1,
    "Sad": 0,
    "Surprised": 0.6,
    "Afraid": 0.1,
    "Disgusted": 0.2,
    "Angry": 0.1,
    "Contempt": 0.3
}

emotions = list(emotions_dict.keys())
dialogue = ""
current_speaker_id = None
emotions = list(emotions_dict.keys())
emotion_scores = []
global_known_faces = {}
global_face_frequency = {}
speaker_id = 0
sentence_cache = []

for i, segment in enumerate(transcript['segments']):
    sentence = segment['text']
    start, end = segment['start'], segment['end']
    interval = (end - start) / 6  # Dividing the interval into parts for 5 frames
    frame_data = [] # generate 5 images of an interval
    
    for j in range(1, 5):  # Looping for 5 frames
        frame_time = start + j * interval

        command = [
            'ffmpeg', '-y', 
            '-ss', str(frame_time),
            '-i', video_path, 
            '-frames:v', '1',
            '-f', 'image2pipe', 
            '-vcodec', 'mjpeg', 
            'pipe:1'
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_data.append(process.stdout)  # Append the frame data to the list

    # identify the speaker
    speaker_id, face_image, annotated_frame = decide_speaker(speaker_id, frame_data, global_known_faces, global_face_frequency)
    
    print(f'\niteration: {i}')
    # speaker in the frame
    if speaker_id:
        emotion = detect_emotions(face_image, emotions)
        timestamp = datetime.now().strftime("%H%M%S")
        face_image.save(f"{annotated_path}/face_image_{timestamp}_{i}.jpg")
    
        if current_speaker_id != speaker_id:
            current_speaker_id = speaker_id
            dialogue += f"\nspeaker{current_speaker_id}({emotion}):"
            print(f"\nspeaker{current_speaker_id}({emotion}):", end="")

        # unfinished sentence
        if sentence_cache:
            stored_sentence = ". ".join(sentence_cache)
            dialogue += stored_sentence
            print(stored_sentence)
            sentence_cache = []

        print(sentence, end="")    
    else:
        sentence_cache.append(sentence)
        
    dialogue += sentence
