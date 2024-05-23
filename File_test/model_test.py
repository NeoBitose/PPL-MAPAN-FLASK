# from flask import Flask
import numpy as np
import pandas as pd
import keras
import json
import requests
from keras import ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers

try:
  res = requests.get('http://127.0.0.1:8000/api/getAllPenyakit')
except:
  print("Dataset tidak bisa dijalankan")
  exit()

data = json.loads(res.text)

# Ubah data menjadi DataFrame
for sample in data:
    sample['gejala'] = ' '.join(sample['gejala'])

# print(sample)
# exit()
    
df = pd.DataFrame(data)
# print(df)
# exit()
# Langkah 2: Preprocessing Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# Label Encoding untuk kolom 'nama_penyakit'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['nama_penyakit'])

# Langkah 3: Bagi Data
# X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Langkah 4-5: Embedding dan Pilih Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_length = len(max(X, key=len))

model = keras.models.load_model('Models/model_test.keras')

predicted_disease = ""
predicted_probability = ""

def predict_disease_with_probability(symptoms):
    # Tokenisasi gejala
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    # Ubah menjadi urutan yang dipadankan
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    # Lakukan prediksi
    prediction = model.predict(symptoms_padded)
    # Hitung probabilitas menggunakan softmax
    probabilities = np.squeeze(np.exp(prediction) / np.sum(np.exp(prediction), axis=1))
    # Decode label penyakit
    labels = label_encoder.classes_
    # Format output sebagai dictionary dengan label dan probabilitasnya
    results = {label: prob for label, prob in zip(labels, probabilities)}
    return results

# Fungsi untuk memprediksi penyakit dengan probabilitas terbesar
def predict_top_disease(symptoms):
    # Tokenisasi gejala
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    # Ubah menjadi urutan yang dipadankan
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    # Lakukan prediksi
    prediction = model.predict(symptoms_padded)
    # Decode label penyakit
    labels = label_encoder.classes_
    # Hitung probabilitas menggunakan softmax
    probabilities = np.squeeze(np.exp(prediction) / np.sum(np.exp(prediction), axis=1))
    # Temukan indeks kelas dengan probabilitas terbesar
    top_class_index = np.argmax(probabilities)
    # Ambil label dan probabilitas kelas teratas
    top_disease = labels[top_class_index]
    top_probability = probabilities[top_class_index]
    return top_disease, top_probability

def pengujian(gejala):
  global predicted_disease, predicted_probability
  input_symptoms = gejala
  predicted_disease, predicted_probability = predict_top_disease(input_symptoms)
  predicted_probabilities = predict_disease_with_probability(input_symptoms)
  predict_prob = []
  return predicted_disease

print(pengujian("Bercak pada tangkai Bercak muda berbentuk bulat kecil berwarna coklat gelap Bercak pada daun berbentuk oval Pada kulit gabah bercak berwarna hitam Ukuran bercak bisa mencapai 1cm"))
print(pengujian("Daun dan pelepah terdapat bercak Bercak daun dan pelepah berbentuk belah ketupat Kehampaan malai padi Tangkai mulai membusuk dan patah Bercak pada daun berwarna keputih-putihan atau keabu-abuan"))
print(pengujian("Banyak anakan menyerupai rumput Daun Sempit Daun Kaku Malai yang dihasikan sedikit bahkan tidak sama sekali Daun bercak berwarna coklat"))
print(pengujian("daun kering bercak pelepah daun dan helai daun gabah tidak penuh tanaman rebah"))
print(pengujian("Pertumbuhan tanaman kerdil Pelepah daun memendek Daun menguning sampai jingga dari pucuk Tanaman menjadi kerdil Daun tua ada bintik-bintik bekas tusukan serangga penular Berkurangnya jumlah anakan"))
print(pengujian("Umumnya menyerang pada tanaman muda 1 sampai 2 minggu Serangan terjadi pada daun yang luka berupa bercak kebasahan Warna bercak hijau keabu-abuan Daun menggulung, mengering warna abu-abu keputihan"))
