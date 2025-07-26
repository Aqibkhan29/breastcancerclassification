import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("cancer_model.pkl")

st.title("🔬 Breast Cancer Predictor")

st.write("Enter the following values based on a patient's diagnosis report:")

# List of all 30 features used in the breast cancer dataset
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Input fields
input_values = []
for feature in features:
    value = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, format="%.5f")
    input_values.append(value)

# Predict button
if st.button("Predict"):
    input_data = np.array([input_values])
    prediction = model.predict(input_data)
    result = "🔴 Malignant" if prediction[0] == 1 else "🟢 Benign"
    st.success(f"Prediction: {result}")
