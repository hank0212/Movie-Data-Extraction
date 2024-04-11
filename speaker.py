from PIL import Image
import numpy as np
from natsort import natsorted
import face_recognition
from io import BytesIO
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
model = load_model(config_path, checkpoint_path)
device = "cpu"

def decide_speaker(speaker_id, frame_data, global_known_faces, global_face_frequency):
    
    local_face_frequency = {}

    # global known faces stores encoding for comparison
    # local known faces stores image for emotion
    local_known_faces = {}

    # loop through frames in 5 intervals
    for frame in frame_data:
        image = Image.open(BytesIO(frame))

        np_image = np.array(image)

        face_locations = face_recognition.face_locations(np_image)
        face_encodings = face_recognition.face_encodings(np_image, face_locations)

        if not face_encodings:
            continue
        
        top, right, bottom, left = face_locations[0]
        
        # Crop the face from the original image and store as tensor
        face_image = image.crop((left, top, right, bottom))
        
        # image transform for DINO instance detection
        transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_source = np.asarray(face_image)
        image_transformed, _ = transform(face_image, None)

        # detect open mouth
        TEXT_PROMPT = "talking"
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.5
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device = "cpu"
        )

        # display annotated image
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        # compare faces
        results = face_recognition.compare_faces(list(global_known_faces.values()), face_encodings[0], tolerance=0.7)

        # if the speaker is known
        if True in results:
            speaker_id = list(global_known_faces.keys())[results.index(True)]
                
        else: # unknown  
            speaker_id = max(global_known_faces) + 1 if global_known_faces else 1 # first iteration
            global_known_faces[speaker_id] = face_encodings[0]

        local_known_faces[speaker_id] = face_image     
        local_face_frequency[speaker_id] = local_face_frequency.get(speaker_id, 0) + 1

        # if the confidence is high
        if logits.nelement() > 0 and (logits[0] > 0.8 or len(phrases) == 2):                  
            return speaker_id, face_image, annotated_frame
            
    if len(local_face_frequency) > 0:
        speaker_id = max(local_face_frequency, key = local_face_frequency.get)
        global_face_frequency[speaker_id] = global_face_frequency.get(speaker_id, 0) + 1
        face_image = local_known_faces[speaker_id]
        return speaker_id, face_image, annotated_frame

    return None, None, None