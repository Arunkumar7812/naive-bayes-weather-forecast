import streamlit as st
import pickle # Changed from joblib to pickle
import pandas as pd
import numpy as np

# 1. Configuration and Model Loading
MODEL_FILENAME = "naive_bayes_weather_model.pkl" # Changed filename
try:
    # Load the model using pickle
    with open(MODEL_FILENAME, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The model file '{MODEL_FILENAME}' was not found. Please ensure you have run your training script to generate this file in the same directory as app.py.")
    st.stop()

# List of all locations used for one-hot encoding during training
# Based on the data structure (20 unique locations)
LOCATIONS = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
    'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
    'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington D.C.'
]

st.title("⛈️ Naive Bayes Weather Forecast")
st.markdown("Enter the weather conditions to predict if it will Rain Tomorrow (1) or not (0).")

# 2. Input Fields for Features
st.subheader("Weather Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider("Temperature (°F)", min_value=30.0, max_value=100.0, value=70.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=65.0, step=0.1)
    wind_speed = st.slider("Wind Speed (mph)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)

with col2:
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=4.0, value=0.5, step=0.01)
    cloud_cover = st.slider("Cloud Cover (%)", min_value=10.0, max_value=100.0, value=50.0, step=0.1)
    pressure = st.slider("Pressure (hPa)", min_value=970.0, max_value=1040.0, value=1010.0, step=0.1)

with col3:
    location = st.selectbox("Location", options=LOCATIONS, index=LOCATIONS.index('Seattle'))

# 3. Prediction Logic
if st.button("Predict Rain Tomorrow"):
    # Create a dictionary for the input features
    input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind Speed': wind_speed,
        'Precipitation': precipitation,
        'Cloud Cover': cloud_cover,
        'Pressure': pressure,
    }
    
    # Initialize all one-hot encoded location columns to 0
    for loc in LOCATIONS:
        input_data[f'Location_{loc}'] = 0
    
    # Set the selected location's one-hot encoded column to 1
    input_data[f'Location_{location}'] = 1

    # Convert the input dictionary to a DataFrame
    # Note: Column order must match the training data!
    feature_order = [
        'Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 
        'Cloud Cover', 'Pressure'
    ] + [f'Location_{loc}' for loc in LOCATIONS]
    
    input_df = pd.DataFrame([input_data], columns=feature_order)

    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The model predicts: **Rain Tomorrow (1)** 🌧️")
    else:
        st.info("The model predicts: **No Rain Tomorrow (0)** ☀️")
