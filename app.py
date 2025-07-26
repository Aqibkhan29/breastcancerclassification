import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("cancer_model.pkl")

st.title("Breast Cancer Predictor")

# Create input fields
radius_mean = st.number_input("Radius Mean", value=0.0)
texture_mean = st.number_input("Texture Mean", value=0.0)
# Add all necessary features...

# When user clicks Predict
if st.button("Predict"):
    features = np.array([[radius_mean, texture_mean]])  # Add more features here
    prediction = model.predict(features)
    st.write("### Result:", "🔴 Malignant" if prediction[0] == 1 else "🟢 Benign")
