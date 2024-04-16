
# Fungsi untuk memprediksi penyakit berdasarkan gejala
import numpy as np
import pandas as pd
import keras
from keras import ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers


# Dataset
data = [
  {
    "nama_penyakit": "Bercak Coklat",
    "gejala": [
      "Bercak pada tangkai",
      "Bercak muda warna coklat gelap ",
      "Bercak daun bentuk oval",
      "kulit gabah bercak warna hitam",
      "Ukuran bercak 1cm"
    ]
  },
  {
    "nama_penyakit": "Penyakit Blast",
    "gejala": [
      "Bercak daun dan pelepah",
      "Bercak daun dan pelepah bentuk belah ketupat",
      "Malai padi tidak ada",
      "Tangkai busuk dan patah",
      "Bercak daun warna putih atau abu-abu"
    ]
  },
  {
    "nama_penyakit": "Kerdil Rumput",
    "gejala": [
      "Anakan seperti rumput",
      "Daun Sempit",
      "Daun Kaku",
      "Malai sedikit atau tidak ada",
      "bercak daun warna coklat"
    ]
  },
  {
    "nama_penyakit": "Hawar Pelepah Daun",
    "gejala": [
      "Daun kering",
      "Bercak pelepah daun dan helai daun",
      "Gabah tidak penuh",
      "Tanaman rebah"
    ]
  },
  {
    "nama_penyakit": "Tungro",
    "gejala": [
      "Tanaman tunbuh kerdil",
      "Pelepah daun pendek",
      "Daun warna kuning atau jingga",
      "Tanaman kerdil",
      "Daun tua bintik-bintik",
      "Anakan kurang"
    ]
  },
  {
    "nama_penyakit": "Kresek",
    "gejala": [
      "menyerang tanaman muda",
      "bercak kebasahan pada daun",
      "bercak warna hijau atau coklat",
      "Daun menggulung dan kering"
    ]
  }
]

# # Menambahkan data tambahan
# additional_data = [
#     {"nama_penyakit": "Bercak Hitam", "gejala": ["Bercak hitam pada tangkai dan daun", "Daun berubah warna menjadi coklat tua", "Bercak muncul pada bagian bawah daun", "Kehilangan produktivitas tanaman"]},
#     {"nama_penyakit": "Hawar Daun Bakteri", "gejala": ["Daun berbintik-bintik berwarna coklat kehitaman", "Bercak menyebar ke seluruh daun", "Bercak tumbuh dan bersatu membentuk garis-garis", "Pengeringan daun terjadi pada kondisi parah"]},
#     {"nama_penyakit": "Penyakit Layu Fusarium", "gejala": ["Tanaman mulai layu dari ujung daun", "Batang berubah warna menjadi coklat", "Tanaman mati secara bertahap", "Akarnya mengalami pembusukan"]},
#     # Tambahkan lebih banyak data penyakit jika diperlukan
# ]

# Menggabungkan data utama dan data tambahan
# data += additional_data

for sample in data:
    sample['gejala'] = ' '.join(sample['gejala'])
# Ubah data menjadi DataFrame
df = pd.DataFrame(data)
print(df)
# Langkah 2: Preprocessing Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# Label Encoding untuk kolom 'nama_penyakit'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['nama_penyakit'])

# Langkah 3: Bagi Data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Langkah 4-5: Embedding dan Pilih Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_length = len(max(X, key=len))

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))  # Regularisasi L2
# model.add(Dropout(0.5))
model.add(Dense(units=len(label_encoder.classes_), activation=relu))  # Menggunakan ReLU sebagai fungsi aktivasi

# Langkah 6-7: Kompilasi Model dan Pelatihan Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Langkah 8: Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Langkah 11: Penyimpanan Model
model.save('D:\codingan\python\github_connect\Sem_4\PPL\plantS_disease_model.h5')
model = keras.models.load_model('D:\codingan\python\github_connect\Sem_4\PPL\plantS_disease_model.h5')

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

# Contoh pengujian
#input_symptoms = "tanaman tunbuh kerdil pelepah daun pendek daun warna kuning atau jingga tanaman kerdil daun tua bintik bintik anakan kurang"
#input_symptoms = "menyerang tanaman muda bercak kebasahan pada daun bercak warna hijau atau coklat daun menggulung dan kering"
input_symptoms = "daun kering bercak pelepah daun dan helai daun gabah tidak penuh tanaman rebah"
predicted_probabilities = predict_disease_with_probability(input_symptoms)
print("Probabilitas Prediksi Penyakit:")
for disease, probability in predicted_probabilities.items():
    print(f"{disease}: {probability}")

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

# Contoh pengujian
predicted_disease, predicted_probability = predict_top_disease(input_symptoms)
print("Prediksi Penyakit Teratas:")
print(f"Penyakit: {predicted_disease}")
print(f"Probabilitas: {predicted_probability}")
