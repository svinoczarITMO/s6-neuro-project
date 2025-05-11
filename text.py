from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd
from pymorphy2 import MorphAnalyzer
from razdel import tokenize

# 1. Предобработка текста для художественных произведений
morph = MorphAnalyzer()

def preprocess_text(text):
    tokens = [token.text for token in tokenize(text)]
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]  # Лемматизация:cite[7]
    return " ".join(lemmas)

# 2. Загрузка и адаптация модели
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=8,  # 8 эмоций по Экману
    id2label={
        0: "гнев",
        1: "отвращение",
        2: "страх",
        3: "радость",
        4: "грусть",
        5: "удивление",
        6: "презрение",
        7: "нейтраль"
    }
)

# 3. Функция предсказания эмоций
def analyze_text_emotion(text):
    processed_text = preprocess_text(text)
    inputs = tokenizer(
        processed_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256,
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {model.config.id2label[i]: round(prob.item(), 3) for i, prob in enumerate(probs[0])}