# Установка библиотек
# !pip install transformers torch

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузка предобученной модели для русского языка
model_name = 'blanchefort/rubert-base-cased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def analyze_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    emotion_labels = ['neutral', 'positive', 'negative']  # Заменить на ваши категории
    return dict(zip(emotion_labels, probabilities.detach().numpy()[0]))