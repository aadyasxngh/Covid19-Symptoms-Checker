"""Prediction logic for COVID-19 using trained ML and DL models."""

# pylint: disable=E0401, E0611, R0911
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
lr_model = joblib.load("models/LR_model")
dt_model = joblib.load("models/DT_model")
rf_model = joblib.load("models/RF_model")
nb_model = joblib.load("models/NB_model")
knn_model = joblib.load("models/KNN_model")
dl_model = load_model("models/DL_model_covid.h5")
scaler = joblib.load("models/scaler.pkl")

def preprocess_input(data):
    """Preprocess user input using saved scaler."""
    return scaler.transform([data])

def predict_with_lr(data):
    """Predict using Logistic Regression model."""
    return lr_model.predict(preprocess_input(data))[0]

def predict_with_dt(data):
    """Predict using Decision Tree model."""
    return dt_model.predict(preprocess_input(data))[0]

def predict_with_rf(data):
    """Predict using Random Forest model."""
    return rf_model.predict(preprocess_input(data))[0]

def predict_with_nb(data):
    """Predict using Naive Bayes model."""
    return nb_model.predict(preprocess_input(data))[0]

def predict_with_knn(data):
    """Predict using K-Nearest Neighbors model."""
    return knn_model.predict(preprocess_input(data))[0]

def predict_with_dl_covid(data):
    """Predict using Deep Learning model."""
    processed = preprocess_input(data)
    result = dl_model.predict(np.array(processed))
    return np.argmax(result, axis=1)[0]

def predict_with_custom_input(symptoms, age, contact, model_name):
    """Prepares input vector and returns prediction from selected model."""
    feature_order = [
        'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
        'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
        'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
        'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe',
        'Contact_Dont-Know', 'Contact_No', 'Contact_Yes'
    ]

    input_vector = [0] * len(feature_order)

    for i, name in enumerate(feature_order[:9]):
        if name in symptoms:
            input_vector[i] = 1

    age_feature = f"Age_{age}"
    if age_feature in feature_order:
        input_vector[feature_order.index(age_feature)] = 1

    input_vector[feature_order.index("Severity_None")] = 1

    contact_map = {
        "Maybe": "Contact_Dont-Know",
        "No": "Contact_No",
        "Yes": "Contact_Yes"
    }
    contact_feature = contact_map.get(contact)
    if contact_feature in feature_order:
        input_vector[feature_order.index(contact_feature)] = 1

    if model_name == "Naive Bayes":
        return predict_with_nb(input_vector)
    if model_name == "Decision Tree":
        return predict_with_dt(input_vector)
    if model_name == "KNN":
        return predict_with_knn(input_vector)
    if model_name == "Random Forest":
        return predict_with_rf(input_vector)
    if model_name == "Logistic Regression":
        return predict_with_lr(input_vector)
    if model_name == "Deep Learning (COVID-19)":
        return predict_with_dl_covid(input_vector)
    return "Unknown model"
