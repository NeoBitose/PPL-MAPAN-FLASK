from flask import Flask
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

app = Flask(__name__)

try:
  res = requests.get('http://127.0.0.1:8000/api/getAllPenyakit')
except:
  print("Dataset tidak bisa dijalankan")
  exit()

data = json.loads(res.text)

# Ubah data menjadi DataFrame
for sample in data:
    sample['gejala'] = ' '.join(sample['gejala'])
    
df = pd.DataFrame(data)
# print(df)
# Langkah 2: Preprocessing Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# Label Encoding untuk kolom 'nama_penyakit'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['nama_penyakit'])

max_length = len(max(X, key=len))
# model = keras.models.load_model('Models\plantS_disease_model.h5')
model = keras.models.load_model('Models\modeltest.keras')

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
  for disease, probability in predicted_probabilities.items():
    dicts = [f"{disease}", f"{probability}"]
    predict_prob.append(dicts)
    # print(f"{disease}: {probability}")  
  obj_predict = [{"name": item[0], "value": item[1]} for item in predict_prob]
  sort_obj_predict = sorted(obj_predict, key=lambda x: float(x["value"]))
  result_diags = [{"name": f"{predicted_disease}", "value": f"{predicted_probability}"}, sort_obj_predict]
  return result_diags

print(pengujian("daun kering bercak pelepah daun dan helai daun gabah tidak penuh tanaman rebah"))

# def prints():
#   return predicted_disease

@app.route("/testing")

def main():
  tes = [1,2,3,4,5]
  return tes 
  return "Model Berhasil Dijalankan..."
  
@app.route("/diagnosa/<gejala>")

def diags(gejala):
  data = pengujian(gejala)
  print(data)
  return data 

if __name__ == "__main__":
    app.run()