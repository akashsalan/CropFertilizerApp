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

    # Pre-filled values
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
    P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=42)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)

    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=22.0, step=0.01)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.01)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=10.0, value=6.5, step=0.01)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0, step=0.01)
    soil_type = st.selectbox("Soil Type", ["sandy", "loamy", "black", "red", "clay"], index=0)

    submitted = st.form_submit_button("Recommend")

if submitted:
    # Crop Prediction
    crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_input_scaled = crop_scaler.transform(crop_input)
    predicted_crop = crop_model.predict(crop_input_scaled)[0]
    st.success(f"✅ Recommended Crop: {predicted_crop.capitalize()}")

    # Handle unsupported crops
    crop_mapping = {
        'rice': 'barley',
        'paddy': 'maize',
        'mothbeans': 'mungbean',
    }
    safe_crop = crop_mapping.get(predicted_crop.lower(), predicted_crop.lower())

    try:
        # Encode crop and soil
        encoded_crop = crop_encoder.transform([safe_crop])[0]
        encoded_soil = soil_encoder.transform([soil_type.lower()])[0]

        # Fertilizer input (with fixed moisture = 50.0)
        fert_input = np.array([[encoded_crop, encoded_soil, N, P, K, temperature, humidity, 50.0]])
        fert_input_scaled = fertilizer_scaler.transform(fert_input)
        predicted_fert = fertilizer_model.predict(fert_input_scaled)[0]
        fert_name = fertilizer_encoder.inverse_transform([predicted_fert])[0]

        # Show fertilizer result
        if safe_crop != predicted_crop.lower():
            st.info(f"""🧪 So... the recommended crop is '{predicted_crop.capitalize()}', but the person who gave me the dataset forgot to include fertilizers for it 😅  
            So here's a fertilizer for a close cousin: '{safe_crop.capitalize()}' 🌱  
            Recommended Fertilizer in {soil_type} soil: {fert_name}""")


        else:
            st.info(f"🧪 Recommended Fertilizer for {safe_crop.capitalize()} in {soil_type} soil: {fert_name}")

    except Exception as e:
        st.warning(f"⚠️ Fertilizer could not be predicted. ({str(e)})")
