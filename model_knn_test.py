import joblib

# Memuat model
model = joblib.load('Models/model_knn.pkl')

# Memuat vectorizer
vectorizer = joblib.load('Models/vectorizer.pkl')

# Fungsi prediksi dengan probabilitas
def predict_penyakit_proba(gejala_list):
    gejala_text = ' '.join(gejala_list)
    gejala_vector = vectorizer.transform([gejala_text])
    proba = model.predict_proba(gejala_vector)[0]
    
    # Mapping penyakit dengan probabilitas
    penyakit_proba = {penyakit: prob for penyakit, prob in zip(model.classes_, proba)}
    
    return penyakit_proba

# Contoh penggunaan fungsi prediksi dengan probabilitas
gejala_baru = [
    "Daun menguning sampai jingga dari pucuk",
    "Tanaman menjadi kerdil",
    "Pelepah daun memendek"
]

penyakit_proba = predict_penyakit_proba(gejala_baru)
print("Probabilitas masing-masing penyakit:")
for penyakit, prob in penyakit_proba.items():
    print(f"{penyakit}: {prob:.2f}")
