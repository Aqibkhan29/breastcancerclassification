import streamlit as st
import numpy as np
import joblib

model = joblib.load("cancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🔬 Breast Cancer Predictor")
st.markdown("Enter values below to predict whether the tumor is malignant or benign.")

# --- 10 example input fields ---
radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0)
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0)

# --- Predict Button ---
if st.button("🔍 Predict"):
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                            fractal_dimension_mean]])
    
    prediction = model.predict(input_data)
    result = "🔴 Malignant" if prediction[0] == 1 else "🟢 Benign"
    st.subheader(f"Prediction: {result}")
