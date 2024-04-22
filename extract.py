import numpy as np
import shutil
import os
import subprocess
from datetime import datetime
import whisper
from speaker import decide_speaker
from emotions import detect_emotions
import warnings
import sys

def extract(video_path, face_image_path, transcript, emotions_dict):

    # delete the annotated path if existed
    if os.path.exists(face_image_path):
        shutil.rmtree(face_image_path)
        os.makedirs(face_image_path, exist_ok=True)

    dialogue = ""
    current_speaker_id = None
    emotions = list(emotions_dict.keys())
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
        
        # speaker in the frame
        if speaker_id:
            emotion = detect_emotions(face_image, emotions)
            timestamp = datetime.now().strftime("%H%M%S")
            face_image.save(f"{face_image_path}/face_image_{timestamp}_{i}.jpg")
        
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
    
    return dialogue 

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':

    video_path = sys.argv[1]
    path_parts = video_path.split('/')
    face_image_path = '/'.join(path_parts[:-1]) + '/face images'
    save_data_path = '/'.join(path_parts[:-1]) + '/data'
    os.makedirs(face_image_path, exist_ok=True)
    os.makedirs(save_data_path, exist_ok=True)

    # transcript from whisper
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

    dialogue = extract(video_path, face_image_path, transcript, emotions_dict)

    timestamp = datetime.now().strftime("%H%M%S")
    os.makedirs(save_data_path, exist_ok=True)
    full_file_path = os.path.join(save_data_path, f"data_{timestamp}")

    with open(full_file_path, 'w') as f:
        f.write(dialogue)
    