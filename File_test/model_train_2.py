import numpy as np
import pandas as pd
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

data = [
    {
        "nama_penyakit": "Bercak Coklat",
        "gejala": [
            "Bercak pada tangkai",
            "Bercak muda berbentuk bulat kecil berwarna coklat gelap",
            "Bercak pada daun berbentuk oval",
            "Pada kulit gabah bercak berwarna hitam",
            "Ukuran bercak bisa mencapai 1cm"
        ]
    },
    {
        "nama_penyakit": "Penyakit Blast",
        "gejala": [
            "Daun dan pelepah terdapat bercak",
            "Bercak daun dan pelepah berbentuk belah ketupat",
            "Kehampaan malai padi",
            "Tangkai mulai membusuk dan patah",
            "Bercak pada daun berwarna keputih-putihan atau keabu-abuan"
        ]
    },
    {
        "nama_penyakit": "Kerdil Rumput",
        "gejala": [
            "Banyak anakan menyerupai rumput",
            "Daun Sempit",
            "Daun Kaku",
            "Malai yang dihasikan sedikit bahkan tidak sama sekali",
            "Daun bercak berwarna coklat"
        ]
    },
    {
        "nama_penyakit": "Hawar Pelepah Padi",
        "gejala": [
            "Daun yang terserang mengering mulai ujung",
            "Tanda bercak pada pelepah daun dan helai daun",
            "Gabah tidak terisi penuh hampa",
            "Tanaman mulai rebah"
        ]
    },
    {
        "nama_penyakit": "Tungro",
        "gejala": [
            "Pertumbuhan tanaman kerdil",
            "Pelepah daun memendek",
            "Daun menguning sampai jingga dari pucuk",
            "Tanaman menjadi kerdil",
            "Daun tua ada bintik-bintik bekas tusukan serangga penular",
            "Berkurangnya jumlah anakan"
        ]
    },
    {
        "nama_penyakit": "Kresek",
        "gejala": [
            "Umumnya menyerang pada tanaman muda 1 sampai 2 minggu",
            "Serangan terjadi pada daun yang luka berupa bercak kebasahan",
            "Warna bercak hijau keabu-abuan",
            "Daun menggulung, mengering warna abu-abu keputihan"
        ]
    }
]

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
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Inisialisasi Parameter Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_length = max(len(seq) for seq in X)
num_classes = len(df['nama_penyakit'].unique())

# Bangun Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(GRU(units=128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(GRU(units=64, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(units=num_classes, activation='softmax'))

# Kompilasi Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Inisialisasi Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih Model
model.fit(X_train, y_train, epochs=15, batch_size=32,
          validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Simpan Model
model.save('Models/model_test.keras')
