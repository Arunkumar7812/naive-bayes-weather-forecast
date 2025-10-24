import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="🌦️ Naïve Bayes Weather Forecast", layout="centered")

st.title("🌦️ Weather Forecast (Naïve Bayes)")
st.write("Predict whether it will **rain tomorrow** based on weather conditions.")

# ----------------------------------------------------
# Load model
# ----------------------------------------------------
MODEL_FILE = "naive_bayes_weather_model.pkl"

model = None
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found! Please upload `weather_nb_model.pkl`.")
    st.stop()

# ----------------------------------------------------
# Input UI
# ----------------------------------------------------
st.header("🌤️ Enter Weather Conditions")

humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=75)
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=28)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=100, value=12)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)
cloud_cover = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, value=60)
location = st.selectbox("🌍 Location", ["Charlotte", "Chicago", "Columbus"])

# ----------------------------------------------------
# Prepare input data
# ----------------------------------------------------
if st.button("🔍 Predict"):
    try:
        input_data = {
            'Humidity': humidity,
            'Temperature': temperature,
            'Wind Speed': wind_speed,
            'Pressure': pressure,
            'Cloud Cover': cloud_cover,
            'Location_Charlotte': 1 if location == "Charlotte" else 0,
            'Location_Chicago': 1 if location == "Chicago" else 0,
            'Location_Columbus': 1 if location == "Columbus" else 0
            'Location_New York': 1 if location == "New York" else 0
        }

        input_df = pd.DataFrame([input_data])

        # Match feature order from model
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------
        prediction = model.predict(input_df)[0]

        st.subheader("🌈 Prediction Result:")
        if prediction == 1:
            st.success("🌧️ It **WILL rain** tomorrow!")
        else:
            st.info("☀️ It **WILL NOT rain** tomorrow.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----------------------------------------------------
# Debug info (optional)
# ----------------------------------------------------
with st.expander("🧩 Debug Info"):
    if hasattr(model, "feature_names_in_"):
        st.write("**Model expects features:**", list(model.feature_names_in_))
    else:
        st.write("Model feature names not stored (trained with NumPy arrays).")
