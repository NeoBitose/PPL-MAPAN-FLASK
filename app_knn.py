import pandas as pd
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from flask import Flask

app = Flask(__name__)

# Dataset
with open('Dataset/data_lengkap.json', 'r') as file:
    data = json.load(file)

# Flattening the dataset
flat_data = []
for item in data:
    for gejala in item['gejala']:
        flat_data.append({'gejala': gejala, 'nama_penyakit': item['nama_penyakit']})

df = pd.DataFrame(flat_data)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['gejala'])

# Label encoding
y = df['nama_penyakit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

def predict_penyakit_proba(gejala_list):
    gejala_text = ' '.join(gejala_list)
    gejala_vector = vectorizer.transform([gejala_text])
    proba = model.predict_proba(gejala_vector)[0]
    penyakit_proba = {penyakit: prob for penyakit, prob in zip(model.classes_, proba)}
    return penyakit_proba

@app.route("/testing")

def main():
  return "Model Berhasil Dijalankan..."

@app.route("/diagnosa/<gejala>")

def diagnosa(gejala):
  print(gejala)
  array = gejala.split(',')
  penyakit_proba = predict_penyakit_proba(array)
  predict_prob = []

  for penyakit, prob in penyakit_proba.items():
    dicts = [f"{penyakit}", f"{prob:.2f}"]
    predict_prob.append(dicts)
  
  obj_predict = [{"name": item[0], "value": item[1]} for item in predict_prob]
  sort_obj_predict = sorted(obj_predict, key=lambda x: float(x["value"]),reverse=True)
  # print(sort_obj_predict)
  result_diags = [{"name": f"{sort_obj_predict[0]['name']}", "value": f"{sort_obj_predict[0]['value']}"}, sort_obj_predict]
  # print(result_diags)
  return result_diags

if __name__ == "__main__":
    app.run()
