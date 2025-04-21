import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === STEP 1: Load & Preprocess Data ===
print("Loading dataset...")
df = pd.read_csv("Cleaned-Data-Final.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')
print("Data scaled and scaler saved.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === STEP 2: Train & Save ML Models ===
print("Training ML models...")
models = {
    "LR_model": LogisticRegression(),
    "DT_model": DecisionTreeClassifier(),
    "RF_model": RandomForestClassifier(),
    "NB_model": GaussianNB(),
    "KNN_model": KNeighborsClassifier()
    # "SVM_model": SVC(probability=True)  ← removed
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}")
    print(f"Saved {name}")

# === STEP 3: Train & Save DL Model ===
print("Training DL model (COVID-19)...")
dl_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')  # assuming binary classification
])

dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

dl_model.save('models/DL_model_covid.h5')
print("DL model saved as DL_model_covid.h5")

print("All models trained and saved successfully!")
