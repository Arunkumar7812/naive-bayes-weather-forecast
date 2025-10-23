import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("weather_nb_model.joblib")  # replace with your model file
st.success("✅ Model loaded successfully.")

st.title("🌦️ Weather Forecast (Naïve Bayes)")

# -----------------------------
# User Inputs
# -----------------------------
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=75)
temperature = st.number_input("Temperature (°C)", min_value=-50, max_value=60, value=28)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=200, value=12)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)
cloud_cover = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, value=50)
location = st.selectbox("Location", ["Charlotte", "Chicago", "Columbus"])

# -----------------------------
# Prepare input dataframe
# -----------------------------
# List of all feature names used during training
feature_names = [
    "Humidity", "Temperature", "Wind Speed", "Pressure", "Cloud Cover",
    "Location_Charlotte", "Location_Chicago", "Location_Columbus"
]

# Initialize all features with 0
input_data = dict.fromkeys(feature_names, 0)

# Fill numerical features
input_data["Humidity"] = humidity
input_data["Temperature"] = temperature
input_data["Wind Speed"] = wind_speed
input_data["Pressure"] = pressure
input_data["Cloud Cover"] = cloud_cover

# Fill location one-hot encoding
location_col = f"Location_{location}"
input_data[location_col] = 1

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# -----------------------------
# Make Prediction
# -----------------------------
if st.button("Predict Rain"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # probability of rain
        st.write(f"🌧️ Prediction: {'Rain' if prediction == 1 else 'No Rain'}")
        st.write(f"💧 Probability of Rain: {probability*100:.2f}%")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
