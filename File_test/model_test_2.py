import numpy as np
import pandas as pd
import keras
import json
import requests
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Mengambil Data dari API
try:
    res = requests.get('http://127.0.0.1:8000/api/getAllPenyakit')
    data = json.loads(res.text)
except Exception as e:
    print(f"Dataset tidak bisa dijalankan: {e}")
    exit()

# Ubah data menjadi DataFrame
for sample in data:
    sample['gejala'] = ' '.join(sample['gejala'])

df = pd.DataFrame(data)

# Tokenisasi dan Padding Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# Label Encoding untuk kolom 'nama_penyakit'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['nama_penyakit'])

# Muat Model yang Sudah Dilatih
model = keras.models.load_model('Models/model_test.keras')

# Parameter Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_length = X.shape[1]  # Menggunakan panjang dari data yang sudah dipadankan

def predict_disease_with_probability(symptoms):
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    prediction = model.predict(symptoms_padded)
    probabilities = np.squeeze(prediction)  # Menggunakan prediksi langsung tanpa perlu eksp dan softmax lagi
    labels = label_encoder.classes_
    results = {label: prob for label, prob in zip(labels, probabilities)}
    return results

def predict_top_disease(symptoms):
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    prediction = model.predict(symptoms_padded)
    probabilities = np.squeeze(prediction)
    labels = label_encoder.classes_
    top_class_index = np.argmax(probabilities)
    top_disease = labels[top_class_index]
    top_probability = probabilities[top_class_index]
    return top_disease, top_probability

def pengujian(gejala):
    predicted_disease, predicted_probability = predict_top_disease(gejala)
    predicted_probabilities = predict_disease_with_probability(gejala)
    return predicted_disease, predicted_probability, predicted_probabilities

# Pengujian
gejala_test = "Bercak pada tangkai Bercak muda berbentuk bulat kecil berwarna coklat gelap Bercak pada daun berbentuk oval Pada kulit gabah bercak berwarna hitam Ukuran bercak bisa mencapai 1cm"
predicted_disease, predicted_probability, predicted_probabilities = pengujian(gejala_test)

print(f"Predicted Disease: {predicted_disease}")
print(f"Predicted Probability: {predicted_probability}")
print("All Predicted Probabilities:")
print(predicted_probabilities)
