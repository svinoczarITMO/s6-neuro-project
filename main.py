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
from emotion_recognition import AudioTransform, EmotionRecognitionModel
from collections import Counter

class EmotionAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор эмоций в речи")
        self.root.geometry("800x600")
        
        # Загрузка модели и encoder
        try:
            # Инициализация модели
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            num_classes = len(self.label_encoder.classes_)
            self.model = EmotionRecognitionModel(num_classes)
            
            # Загрузка весов
            self.model.load_state_dict(torch.load('models/emotion_recognition_model.pth'))
            self.model.eval()  # Переключение в режим оценки
            
            # Инициализация преобразования аудио
            self.audio_transform = AudioTransform(target_length=16000)
            
            print("Модель и encoder успешно загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель или encoder: {str(e)}")
            root.destroy()
            return

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для загрузки файла
        upload_frame = ttk.LabelFrame(self.root, text="Загрузка аудио", padding="10")
        upload_frame.pack(fill="x", padx=10, pady=5)

        self.file_path = tk.StringVar()
        ttk.Entry(upload_frame, textvariable=self.file_path, width=50).pack(side="left", padx=5)
        ttk.Button(upload_frame, text="Выбрать файл", command=self.load_file).pack(side="left", padx=5)
        ttk.Button(upload_frame, text="Анализировать", command=self.analyze_audio).pack(side="left", padx=5)

        # Фрейм для результатов
        results_frame = ttk.LabelFrame(self.root, text="Результаты анализа", padding="10")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Текстовое поле для результатов
        self.results_text = tk.Text(results_frame, height=10, width=60)
        self.results_text.pack(fill="both", expand=True)

        # Фрейм для графика
        self.plot_frame = ttk.LabelFrame(self.root, text="Визуализация", padding="10")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)

    def analyze_audio(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showwarning("Предупреждение", "Выберите аудиофайл для анализа")
            return

        try:
            # Загрузка и обработка аудио
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Конвертация в моно если стерео
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Список для хранения предсказаний
            predictions = []
            probabilities_list = []
            
            # Делаем 10 предсказаний
            for _ in range(10):
                # Применение преобразований
                mel_spec = self.audio_transform(waveform)
                
                # Добавление размерности батча
                mel_spec = mel_spec.unsqueeze(0)
                
                # Получение предсказания
                with torch.no_grad():
                    output = self.model(mel_spec)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                    predictions.append(predicted_class)
                    probabilities_list.append(probabilities)
            
            # Находим наиболее часто встречающийся класс
            most_common_class = Counter(predictions).most_common(1)[0][0]
            
            # Вычисляем средние вероятности
            avg_probabilities = torch.stack(probabilities_list).mean(dim=0)
            confidence = avg_probabilities[most_common_class].item()

            # Получение названия эмоции
            emotion = self.label_encoder.inverse_transform([most_common_class])[0]

            # Отображение результатов
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Файл: {os.path.basename(file_path)}\n")
            self.results_text.insert(tk.END, f"Определенная эмоция: {emotion}\n")
            self.results_text.insert(tk.END, f"Уверенность: {confidence:.2%}\n\n")
            
            # Отображение распределения предсказаний
            self.results_text.insert(tk.END, "Распределение предсказаний:\n")
            for class_idx, count in Counter(predictions).most_common():
                emotion_name = self.label_encoder.inverse_transform([class_idx])[0]
                self.results_text.insert(tk.END, f"{emotion_name}: {count}/10 раз\n")
            
            self.results_text.insert(tk.END, "\nСредние вероятности:\n")
            for i, prob in enumerate(avg_probabilities):
                emotion_name = self.label_encoder.inverse_transform([i])[0]
                self.results_text.insert(tk.END, f"{emotion_name}: {prob:.2%}\n")

            # Создание графика
            self.create_plot(avg_probabilities.numpy(), self.label_encoder.classes_)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при анализе: {str(e)}")

    def create_plot(self, probabilities, emotions):
        # Очистка предыдущего графика
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Создание нового графика
        plt.figure(figsize=(8, 4))
        plt.bar(emotions, probabilities)
        plt.title("Распределение вероятностей эмоций")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохранение графика в буфер
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Отображение графика в интерфейсе
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(self.plot_frame, image=photo)
        label.image = photo
        label.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()