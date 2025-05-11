import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Создаем и обучаем LabelEncoder
label_encoder = LabelEncoder()
emotions = ['angry', 'neutral', 'sad', 'happy', 'fear']
label_encoder.fit(emotions)

# Сохраняем encoder
joblib.dump(label_encoder, 'label_encoder.joblib')
print("LabelEncoder успешно сохранен в 'label_encoder.joblib'") 