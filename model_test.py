import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load dataset for obtaining tokenizer and label encoder
df = pd.read_json('Dataset\data.json')

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gejala'])
X = tokenizer.texts_to_sequences(df['gejala'])
X = pad_sequences(X)

# Label Encoding for the 'nama_penyakit' column
label_encoder = LabelEncoder()
label_encoder.fit(df['nama_penyakit'])

# Load the trained model
model = load_model('Models/model_test.keras')

def predict_disease(symptoms):
    # Tokenize symptoms
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    # Pad the sequence
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=len(max(X, key=len)))
    # Perform prediction
    prediction = model.predict(symptoms_padded)
    # Calculate probabilities using softmax
    probabilities = np.squeeze(np.exp(prediction) / np.sum(np.exp(prediction), axis=1))
    # Decode disease labels
    labels = label_encoder.classes_
    # Format output as a dictionary with disease labels and their probabilities
    results = {label: prob for label, prob in zip(labels, probabilities)}
    return results

def predict_top_disease(symptoms):
    # Tokenize symptoms
    symptoms_sequence = tokenizer.texts_to_sequences([symptoms])
    # Pad the sequence
    symptoms_padded = pad_sequences(symptoms_sequence, maxlen=len(max(X, key=len)))
    # Perform prediction
    prediction = model.predict(symptoms_padded)
    # Decode disease labels
    labels = label_encoder.classes_
    # Calculate probabilities using softmax
    probabilities = np.squeeze(np.exp(prediction) / np.sum(np.exp(prediction), axis=1))
    # Find the index of the class with the highest probability
    top_class_index = np.argmax(probabilities)
    # Get the top disease label and probability
    top_disease = labels[top_class_index]
    top_probability = probabilities[top_class_index]
    return top_disease, top_probability

def testing(symptoms):
    predicted_disease = predict_top_disease(symptoms)
    predicted_probabilities = predict_disease(symptoms)
    return predicted_disease, predicted_probabilities

# Example testing
symptoms = "Bercak pada tangkai Bercak muda berbentuk bulat kecil berwarna coklat gelap Bercak pada daun berbentuk oval Pada kulit gabah bercak berwarna hitam Ukuran bercak bisa mencapai 1cm"
predicted_disease, predicted_probabilities = testing(symptoms)

print(f"Predicted Disease: {predicted_disease}")
print(f"Predicted Probabilities: {predicted_probabilities}")
