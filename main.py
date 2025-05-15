import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
from voice_recognition import AudioTransform, EmotionRecognitionModel
from collections import Counter
from moviepy.editor import VideoFileClip
from face_recognition import process_video
# import logging
from loguru import logger
import threading
import cv2
import time
import tempfile
import shutil
from speech_to_text import analyze_audio

# Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class EmotionAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор эмоций")
        self.root.geometry("1000x800")

        # Загрузка модели и encoder
        try:
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            num_classes = len(self.label_encoder.classes_)
            self.audio_model = EmotionRecognitionModel(num_classes)
            self.audio_model.load_state_dict(torch.load('models/emotion_recognition_model.pth'))
            self.audio_model.eval()
            self.audio_transform = AudioTransform(target_length=16000)

            logger.info("Модели успешно загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модели: {str(e)}")
            root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        upload_frame = ttk.LabelFrame(self.root, text="Загрузка файла", padding="10")
        upload_frame.pack(fill="x", padx=10, pady=5)

        self.file_path = tk.StringVar()
        ttk.Entry(upload_frame, textvariable=self.file_path, width=50).pack(side="left", padx=5)
        ttk.Button(upload_frame, text="Выбрать файл", command=self.load_file).pack(side="left", padx=5)
        ttk.Button(upload_frame, text="Анализировать", command=self.analyze_file).pack(side="left", padx=5)

        results_frame = ttk.LabelFrame(self.root, text="Результаты анализа", padding="10")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, height=10, width=60)
        self.results_text.pack(fill="both", expand=True)

        self.plot_frame = ttk.LabelFrame(self.root, text="Визуализация", padding="10")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.progress_frame = ttk.LabelFrame(self.root, text="Прогресс", padding="10")
        self.progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(fill="x", padx=5, pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mov"), ("Audio files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)

    def extract_audio_from_video(self, video_path, audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()

    def analyze_audio_segment(self, audio_path, segment_duration):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Передискретизация на 16000 Гц
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        mel_spec = self.audio_transform(waveform)
        mel_spec = mel_spec.unsqueeze(0)

        with torch.no_grad():
            output = self.audio_model(mel_spec)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]

        logger.info(f"Аудио сегмент: эмоция - {emotion}")

        return emotion

    def analyze_audio_emotions(self, audio_path, segment_duration):
        try:
            logger.info('Анализ аудио...')
            waveform, sample_rate = torchaudio.load(audio_path)
            total_samples = len(waveform[0])
            segment_samples = int(segment_duration * sample_rate)
            num_segments = total_samples // segment_samples

            segment_emotions = []
            temp_dir = tempfile.mkdtemp()

            for i in range(num_segments):
                logger.info(f'Анализ аудио сегмента {i}...')
                segment = waveform[:, i * segment_samples:(i + 1) * segment_samples]
                segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
                torchaudio.save(segment_path, segment, sample_rate)
                emotion = self.analyze_audio_segment(segment_path, segment_duration)
                segment_emotions.append(emotion)

            shutil.rmtree(temp_dir)
            return segment_emotions

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при анализе аудио: {str(e)}")
            return None

    def analyze_file(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showwarning("Предупреждение", "Выберите файл для анализа")
            return

        def analyze():
            try:
                if file_path.endswith(('.mp4', '.mov', '.MOV')):
                    audio_path = "temp_audio.wav"
                    self.extract_audio_from_video(file_path, audio_path)

                    video = cv2.VideoCapture(file_path)
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps
                    video.release()

                    segment_duration = 2.0  # Длительность сегмента в секундах
                    audio_emotions = self.analyze_audio_emotions(audio_path, segment_duration)
                    transcriptions = analyze_audio(audio_path)

                    self.progress_bar["value"] = 0
                    self.progress_bar["maximum"] = 100
                    self.root.update_idletasks()

                    def update_progress(current, total):
                        progress = (current / total) * 100
                        self.progress_bar["value"] = progress
                        self.root.update_idletasks()

                    video_results = process_video(file_path, frames_per_second=10, batch_size=10, progress_callback=update_progress)
                    logger.info(f'VIDEO RESULTS: {video_results}')

                    self.root.after(0, lambda: self.update_results(file_path, audio_emotions, transcriptions, video_results))

                elif file_path.endswith('.wav'):
                    audio_emotions = self.analyze_audio_emotions(file_path, segment_duration=2.0)
                    transcriptions = analyze_audio(file_path)

                    self.root.after(0, lambda: self.update_results(file_path, audio_emotions, transcriptions))

            except Exception as e:
                messagebox.showerror("Ошибка", f"Произошла ошибка при анализе: {str(e)}")

        threading.Thread(target=analyze).start()

    def update_results(self, file_path, audio_emotions, transcriptions, video_results=None):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Файл: {os.path.basename(file_path)}\n")

        if video_results:
            for i, (audio_emotion, (start, end, transcription), video_result) in enumerate(zip(audio_emotions, transcriptions, video_results)):
                self.results_text.insert(tk.END, f"[{time.strftime('%M:%S', time.gmtime(start))}.{int((start % 1) * 1000):03d} - {time.strftime('%M:%S', time.gmtime(end))}.{int((end % 1) * 1000):03d}]: аудио - {audio_emotion}, видео - {video_result.get('dominant_emotion', 'N/A')}, текст - {transcription}\n")

                logger.info(f"[{time.strftime('%M:%S', time.gmtime(start))}.{int((start % 1) * 1000):03d} - {time.strftime('%M:%S', time.gmtime(end))}.{int((end % 1) * 1000):03d}]: аудио - {audio_emotion}, видео - {video_result.get('dominant_emotion', 'N/A')}, текст - {transcription}")

            self.create_plot(audio_emotions, video_results)
        else:
            for i, (emotion, (start, end, transcription)) in enumerate(zip(audio_emotions, transcriptions)):
                self.results_text.insert(tk.END, f"[{time.strftime('%M:%S', time.gmtime(start))}.{int((start % 1) * 1000):03d} - {time.strftime('%M:%S', time.gmtime(end))}.{int((end % 1) * 1000):03d}]: аудио - {emotion}, текст - {transcription}\n")

                logger.info(f"[{time.strftime('%M:%S', time.gmtime(start))}.{int((start % 1) * 1000):03d} - {time.strftime('%M:%S', time.gmtime(end))}.{int((end % 1) * 1000):03d}]: аудио - {emotion}, текст - {transcription}")

            self.create_plot(audio_emotions)

    def create_plot(self, audio_emotions, video_results=None):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # График для аудио
        axs[0].bar(range(len(audio_emotions)), audio_emotions)
        axs[0].set_title("Распределение эмоций по сегментам (аудио)")
        axs[0].set_xticks(range(len(audio_emotions)))
        axs[0].set_xticklabels([f"{i*2}-{(i+1)*2}s" for i in range(len(audio_emotions))], rotation=45)

        # График для видео
        if video_results:
            video_emotions = [result.get('dominant_emotion', 'N/A') for result in video_results]
            axs[1].bar(range(len(video_emotions)), video_emotions)
            axs[1].set_title("Распределение эмоций по сегментам (видео)")
            axs[1].set_xticks(range(len(video_emotions)))
            axs[1].set_xticklabels([f"{i*2}-{(i+1)*2}s" for i in range(len(video_emotions))], rotation=45)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(self.plot_frame, image=photo)
        label.image = photo
        label.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()
