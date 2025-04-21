"""Streamlit frontend for COVID-19 symptoms checker."""

import streamlit as st
from ml_dl_predict import predict_with_custom_input

# --- Simulated user store (in-memory) ---
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_form():
    """Display login and signup form."""
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username in st.session_state.users and \
               st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Logged in successfully.")
            else:
                st.error("❌ Invalid username or password.")

    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.warning("⚠️ Username already exists.")
            elif new_user and new_pass:
                st.session_state.users[new_user] = new_pass
                st.success("✅ Account created! Please log in.")
            else:
                st.warning("⚠️ Please enter both username and password.")

def main_app():
    """Main app interface for symptom input and prediction."""
    st.markdown(
        "<h1 style='text-align: center;'>🦠 Covid-19 Symptoms Checker</h1>",
        unsafe_allow_html=True
    )

    symptom_options = [
        "Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing",
        "Sore-Throat", "Pains", "Nasal-Congestion", "Runny-Nose", "Diarrhea"
    ]
    symptoms = st.multiselect("Select your symptoms (if any):", symptom_options)

    age_options = ["0-9", "10-19", "20-24", "25-59", "60+"]
    age = st.radio("Select your age group:", age_options, index=None)

    contact_options = ["Yes", "No", "Maybe"]
    contact = st.radio("Had contact with a COVID-positive person?", contact_options, index=None)

    prediction_type = st.selectbox("Choose Prediction Type:", ["ML Models", "DL Model"])

    model_name = None
    if prediction_type == "ML Models":
        model_options = [
            "Naive Bayes", "Decision Tree", "KNN", "Random Forest", "Logistic Regression"
        ]
        model_name = st.selectbox("Choose ML Model:", model_options)
    elif prediction_type == "DL Model":
        model_options = ["Deep Learning (COVID-19)"]
        model_name = st.selectbox("Choose DL Model:", model_options)

    if st.button("Check"):
        if age is None or contact is None or model_name is None:
            st.warning("⚠️ Please fill out all details.")
        else:
            prediction = predict_with_custom_input(symptoms, age, contact, model_name)
            if prediction == 1:
                st.error("⚠️ You are likely infected with COVID-19.")
            else:
                st.success("✅ You are unlikely to be infected with COVID-19.")

if not st.session_state.logged_in:
    login_form()
else:
    main_app()
