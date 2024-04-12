# Installation

```python
pip install -r requirements.txt
```

# Usage
``` python
python extract.py <path to video source>
```

file structure:
MOIVE-DATA-EXTRACTION/
|--Videos
|   |-- Video1 Folder
|   |   |--video1.mp4
|   |-- Video2 Folder
|   |   |--video2.mp4
|--GroundingDINO
|-- extract.py
|-- speaker.py
|-- emotions.py
|-- README.md
|-- requirements.txt


# Notes
- The face images folder has the detected face for user to manually categorize facial expression if the detection went wrong.
- The index is associated with the transcript index.
- Generated data is stored in the video folder, along with the face images folder
- Accuracy is around 70%, some adjustment is needed.



