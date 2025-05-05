# Установка библиотек
# !pip install deepface opencv-python

from deepface import DeepFace
import cv2

def analyze_video_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotions.append(result[0]['dominant_emotion'])
    cap.release()
    return max(set(emotions), key=emotions.count)