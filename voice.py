# Установка библиотек
# !pip install librosa tensorflow

import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка предобученной модели на RAVDESS
audio_model = load_model('audio_emotion_model.h5')  # Модель требует обучения

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def analyze_audio_emotion(audio_path):
    features = extract_audio_features(audio_path)
    prediction = audio_model.predict(features.reshape(1, -1))
    return prediction[0]