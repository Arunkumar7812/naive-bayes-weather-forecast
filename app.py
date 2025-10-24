import streamlit as st
import pickle 
import pandas as pd
import numpy as np # Needed for log transformation

# 1. Configuration and Model Loading
MODEL_FILENAME = "naive_bayes_weather_model.pkl"

# Full list of locations. MUST be sorted alphabetically to match pandas OHE column creation with drop_first=False.
ALL_LOCATIONS_SORTED = [
    'Austin', 'Charlotte', 'Chicago', 'Columbus', 'Dallas', 'Denver', 
    'Fort Worth', 'Houston', 'Indianapolis', 'Jacksonville', 'Los Angeles', 
    'New York', 'Philadelphia', 'Phoenix', 'San Antonio', 'San Diego', 
    'San Francisco', 'San Jose', 'Seattle', 'Washington D.C.'
]

# The list of ALL 20 ONE-HOT ENCODED COLUMNS expected by the model (since Location_Austin was expected).
OHE_LOCATIONS_EXPECTED = ALL_LOCATIONS_SORTED

# Define the EXACT feature order used in training
FEATURE_ORDER_NUMERICAL = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation_log', # Note the change here
    'Cloud Cover', 'Pressure'
]

# The OHE columns must be listed exactly as the model expects them:
FEATURE_ORDER_OHE = [f'Location_{loc}' for loc in OHE_LOCATIONS_EXPECTED]

FINAL_FEATURE_ORDER = FEATURE_ORDER_NUMERICAL + FEATURE_ORDER_OHE

try:
    # Load the model using pickle
    with open(MODEL_FILENAME, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The model file '{MODEL_FILENAME}' was not found. Please ensure you have run your training script to generate this file in the same directory as app.py.")
    st.stop()


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
    # Get the original precipitation value from the user
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=4.0, value=0.5, step=0.01) 
    cloud_cover = st.slider("Cloud Cover (%)", min_value=10.0, max_value=100.0, value=50.0, step=0.1)
    pressure = st.slider("Pressure (hPa)", min_value=970.0, max_value=1040.0, value=1010.0, step=0.1)

with col3:
    location = st.selectbox("Location", options=ALL_LOCATIONS_SORTED, index=ALL_LOCATIONS_SORTED.index('Seattle'))

# 3. Prediction Logic
if st.button("Predict Rain Tomorrow"):
    # 1. Apply log transformation to the precipitation input
    # Use log(1+x) (np.log1p) to handle 0 precipitation values gracefully
    precipitation_log = np.log1p(precipitation) 
    
    # 2. Collect all features
    input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind Speed': wind_speed,
        'Cloud Cover': cloud_cover,
        'Pressure': pressure,
        'Precipitation_log': precipitation_log, # Include the new log-transformed feature
    }
    
    # 3. Initialize ALL 20 OHE columns to 0
    for ohe_col in FEATURE_ORDER_OHE:
        # Extract the location name (e.g., 'Location_Austin' -> 'Austin')
        loc_name = ohe_col.replace('Location_', '')
        
        # Set to 1 only for the selected location
        input_data[ohe_col] = 1 if loc_name == location else 0

    # 4. Convert to DataFrame using the EXACT training feature order
    input_df = pd.DataFrame([input_data], columns=FINAL_FEATURE_ORDER)

    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The model predicts: **Rain Tomorrow (1)** 🌧️")
    else:
        st.info("The model predicts: **No Rain Tomorrow (0)** ☀️")
