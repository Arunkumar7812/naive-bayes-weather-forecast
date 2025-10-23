import streamlit as st
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB

# -------------------------------
# 1️⃣ App Configuration
# -------------------------------
st.set_page_config(page_title="Weather Forecast - Naive Bayes", layout="centered")
st.title("🌦️ Weather Forecast (Naïve Bayes)")
st.write("Predict whether it will rain tomorrow based on humidity, temperature, wind speed, and pressure.")

# -------------------------------
# 2️⃣ Load Trained Model
# -------------------------------
MODEL_FILE = "naive_bayes_weather_model.pkl"

try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model file: {e}")
    st.stop()

# -------------------------------
# 3️⃣ User Inputs
# -------------------------------
st.header("Enter Weather Conditions")

col1, col2 = st.columns(2)
with col1:
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=75)
    temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=60, value=28)
with col2:
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=200, value=12)
    pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)

# -------------------------------
# 4️⃣ Predict Button
# -------------------------------
if st.button("🔍 Predict"):
    try:
        input_data = pd.DataFrame({
            "humidity": [humidity],
            "temperature": [temperature],
            "wind_speed": [wind_speed],
            "pressure": [pressure]
        })
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("🌧️ Prediction: It **WILL rain** tomorrow!")
        else:
            st.info("☀️ Prediction: It **WILL NOT rain** tomorrow.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------
# 5️⃣ Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Arun Kumar S | Naïve Bayes Weather Prediction")
