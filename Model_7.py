import numpy as np
import pandas as pd
import keras
import json 
import requests
from keras import ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Dataset
# data = [
#   {
#     "nama_penyakit": "Bercak Coklat",
#     "gejala": [
#       "Bercak pada tangkai",
#       "Bercak muda warna coklat gelap ",
#       "Bercak daun bentuk oval",
#       "kulit gabah bercak warna hitam",
#       "Ukuran bercak 1cm"
#     ]
#   },
#   {
#     "nama_penyakit": "Penyakit Blast",
#     "gejala": [
#       "Bercak daun dan pelepah",
#       "Bercak daun dan pelepah bentuk belah ketupat",
#       "Malai padi tidak ada",
#       "Tangkai busuk dan patah",
#       "Bercak daun warna putih atau abu-abu"
#     ]
#   },
#   {
#     "nama_penyakit": "Kerdil Rumput",
#     "gejala": [
#       "Anakan seperti rumput",
#       "Daun Sempit",
#       "Daun Kaku",
#       "Malai sedikit atau tidak ada",
#       "bercak daun warna coklat"
#     ]
#   },
#   {
#     "nama_penyakit": "Hawar Pelepah Daun",
#     "gejala": [
#       "Daun kering",
#       "Bercak pelepah daun dan helai daun",
#       "Gabah tidak penuh",
#       "Tanaman rebah"
#     ]
#   },
#   {
#     "nama_penyakit": "Tungro",
#     "gejala": [
#       "Tanaman tunbuh kerdil",
#       "Pelepah daun pendek",
#       "Daun warna kuning atau jingga",
#       "Tanaman kerdil",
#       "Daun tua bintik-bintik",
#       "Anakan kurang"
#     ]
#   },
#   {
#     "nama_penyakit": "Kresek",
#     "gejala": [
#       "menyerang tanaman muda",
#       "bercak kebasahan pada daun",
#       "bercak warna hijau atau coklat",
#       "Daun menggulung dan kering"
#     ]
#   }
# ]
# data = [
#   {
#     "nama_penyakit": "Bercak Coklat",
#     "gejala": [
#       "Bercak pada tangkai",
#       "Bercak muda warna coklat gelap",
#       "Bercak daun bentuk oval",
#       "Kulit gabah bercak warna hitam",
#       "Ukuran bercak 1cm",
#       "Daun kering",
#       "Pelepah daun dan helai daun berwarna kuning",
#       "Tangkai busuk dan patah",
#       "Pada malai padi terdapat bercak kehitaman"
#     ]
#   },
#   {
#     "nama_penyakit": "Penyakit Blast",
#     "gejala": [
#       "Bercak daun dan pelepah",
#       "Bercak daun dan pelepah bentuk belah ketupat",
#       "Malai padi tidak ada",
#       "Tangkai busuk dan patah",
#       "Bercak daun warna putih atau abu-abu"
#     ]
#   },
#   {
#     "nama_penyakit": "Kerdil Rumput",
#     "gejala": [
#       "Anakan seperti rumput",
#       "Daun sempit dan kaku",
#       "Malai sedikit atau tidak ada",
#       "Bercak daun warna coklat"
#     ]
#   },
#   {
#     "nama_penyakit": "Hawar Pelepah Daun",
#     "gejala": [
#       "Daun kering",
#       "Bercak pelepah daun dan helai daun",
#       "Gabah tidak penuh",
#       "Tanaman rebah"
#     ]
#   },
#   {
#     "nama_penyakit": "Tungro",
#     "gejala": [
#       "Tanaman tunbuh kerdil",
#       "Pelepah daun pendek",
#       "Daun warna kuning atau jingga",
#       "Tanaman kerdil",
#       "Daun tua bintik-bintik",
#       "Anakan kurang"
#     ]
#   },
#   {
#     "nama_penyakit": "Kresek",
#     "gejala": [
#       "Menyerang tanaman muda",
#       "Bercak kebasahan pada daun",
#       "Bercak warna hijau atau coklat",
#       "Daun menggulung dan kering"
#     ]
#   }
# ]
res = requests.get('http://127.0.0.1:8000/api/getAllPenyakit')
data = json.loads(res.text)

# Ubah data menjadi DataFrame
df = pd.DataFrame(data)

# Tokenisasi dan Padding Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# One-hot Encoding untuk kolom 'nama_penyakit'
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(df['nama_penyakit'].values.reshape(-1, 1))

# Bagi Data menjadi Data Latih, Validasi, dan Uji
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Inisialisasi Parameter Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_length = len(max(X, key=len))
num_classes = len(df['nama_penyakit'].unique())

# Bangun Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(units=num_classes, activation='softmax'))

# Kompilasi Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih Model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Fungsi Prediksi dengan Probabilitas
def predict_disease_with_probability(symptoms):
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    prediction = model.predict(symptoms_padded)
    probabilities = np.squeeze(prediction)
    labels = onehot_encoder.categories_[0]
    results = {label: prob for label, prob in zip(labels, probabilities)}
    return results

model.save('plantS_disease_model.h5')
model = keras.models.load_model('plantS_disease_model.h5')

# Prediksi Penyakit Teratas
def predict_top_disease(symptoms):
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=max_length)
    prediction = model.predict(symptoms_padded)
    label_index = np.argmax(prediction)
    disease = onehot_encoder.categories_[0][label_index]
    probability = prediction[0][label_index]
    return disease, probability

# Contoh pengujian
input_symptoms = "Tanaman tumbuh kerdil Pelepah daun pendek Daun warna kuning atau jingga"
predicted_probabilities = predict_disease_with_probability(input_symptoms)
print("Probabilitas Prediksi Penyakit:")
for disease, probability in predicted_probabilities.items():
    print(f"{disease}: {probability}")

predicted_disease, predicted_probability = predict_top_disease(input_symptoms)
print("Prediksi Penyakit Teratas:")
print(f"Penyakit: {predicted_disease}")
print(f"Probabilitas: {predicted_probability}")
