import clip
import torch
import numpy as np


'''
return the emotion from the facial image

Parameters:
image(str): image of the face
emotions(list): list of emotion, default is 7 emotions

Returns:
emotion(str): the most likely emotion from the image

'''
def detect_emotions(image, emotions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(emotions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return emotions[np.argmax(probs)]