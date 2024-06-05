import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Membaca data dari file JSON
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

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculating loss (using Mean Squared Error as an example)
# First, convert the class labels to numerical values for MSE calculation
y_test_numeric = pd.factorize(y_test)[0]
y_pred_numeric = pd.factorize(y_pred)[0]
mse = mean_squared_error(y_test_numeric, y_pred_numeric)
print(f"Mean Squared Error: {mse:.2f}")

# Menyimpan model dan vectorizer
joblib.dump(model, 'Models/model_knn.pkl')
joblib.dump(vectorizer, 'Models/vectorizer.pkl')
