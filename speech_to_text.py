import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from pydub.silence import split_on_silence
from loguru import logger

# Загрузка модели
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

def split_audio(audio_path, min_silence_len=500, silence_thresh=-40, keep_silence=200):
    """Разбиваем аудио на фрагменты по тишине"""
    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    return chunks

def process_chunk(chunk, chunk_idx):
    """Обработка одного аудиофрагмента"""
    # Экспортируем фрагмент во временный файл
    chunk.export(f"temp_chunk_{chunk_idx}.wav", format="wav")

    # Загружаем как массив numpy
    speech_array, sr = librosa.load(f"temp_chunk_{chunk_idx}.wav", sr=16_000)

    # Обработка через модель
    inputs = processor(speech_array, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def analyze_audio(audio_path):
    """Основная функция анализа"""
    print(f"\nAnalyzing: {audio_path}")

    # Разбиваем на фрагменты
    chunks = split_audio(audio_path)
    print(f"Found {len(chunks)} chunks")

    # Обрабатываем каждый фрагмент
    full_text = []
    start_time = 0.0
    for i, chunk in enumerate(chunks):
        try:
            text = process_chunk(chunk, i)
            end_time = start_time + len(chunk) / 212.0  # Длительность чанка в секундах
            print(f"Chunk {i+1}: {text}")
            full_text.append((start_time, end_time, text))
            start_time = end_time
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            end_time = start_time + len(chunk) / 212.0  # Длительность чанка в секундах
            full_text.append((start_time, end_time, ""))
            start_time = end_time

    return full_text
