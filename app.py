# app.py
import streamlit as st
import numpy as np
import pickle

# Load crop model and scaler
crop_model = pickle.load(open("crop_model.sav", "rb"))
crop_scaler = pickle.load(open("crop_scaler.sav", "rb"))

# Load fertilizer model and scaler
fertilizer_model = pickle.load(open("fertilizer_model.sav", "rb"))
fertilizer_scaler = pickle.load(open("fertilizer_scaler.sav", "rb"))

# Load encoders
crop_encoder = pickle.load(open("crop_encoder.sav", "rb"))
soil_encoder = pickle.load(open("soil_encoder.sav", "rb"))
fertilizer_encoder = pickle.load(open("fertilizer_encoder.sav", "rb"))

st.set_page_config(page_title="Crop & Fertilizer Recommender")

st.title("🌾 Crop and Fertilizer Recommendation System")

# Input form
with st.form("input_form"):
    st.header("🌱 Enter Soil & Environmental Details")
    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorous (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    humidity = st.slider("Humidity (%)", 20.0, 100.0, 60.0)
    ph = st.slider("Soil pH", 3.0, 10.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
    soil_type = st.selectbox("Soil Type", ["sandy", "loamy", "black", "red", "clay"])

    submitted = st.form_submit_button("Recommend")

if submitted:
    # Crop Prediction
    crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_input_scaled = crop_scaler.transform(crop_input)
    predicted_crop = crop_model.predict(crop_input_scaled)[0]
    st.success(f"✅ Recommended Crop: {predicted_crop.capitalize()}")

    # Fertilizer Prediction
    try:
        encoded_crop = crop_encoder.transform([predicted_crop.lower()])[0]
        encoded_soil = soil_encoder.transform([soil_type.lower()])[0]

        fert_input = np.array([[encoded_crop, encoded_soil, N, P, K, temperature, humidity, 50.0]])
        fert_input_scaled = fertilizer_scaler.transform(fert_input)
        predicted_fert = fertilizer_model.predict(fert_input_scaled)[0]
        fert_name = fertilizer_encoder.inverse_transform([predicted_fert])[0]

        st.info(f"🧪 Recommended Fertilizer: {fert_name}")
    except Exception as e:
        st.warning("⚠️ Fertilizer could not be predicted for this crop.")
