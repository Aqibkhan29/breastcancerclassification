import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

model = joblib.load("cancer_model.pkl")
st.title("🔬 Breast Cancer Predictor")

st.write("Choose how you'd like to input patient data:")

option = st.radio("Select input method:", ("Manual Entry", "Upload CSV"))

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Option 1: Manual entry
if option == "Manual Entry":
    st.subheader("Enter values for 30 features:")
    input_values = []
    for feature in features:
        val = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, format="%.5f")
        input_values.append(val)

    if st.button("Predict"):
        input_data = np.array([input_values])
        prediction = model.predict(input_data)
        result = "🔴 Malignant" if prediction[0] == 1 else "🟢 Benign"
        st.success(f"Prediction: {result}")

# Option 2: Upload CSV
else:
    uploaded_file = st.file_uploader("Upload a CSV file with the 30 features", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📊 Uploaded Data Preview:", df.head())
            
            if all(feature in df.columns for feature in features):
                predictions = model.predict(df[features])
                df["Prediction"] = ["Malignant 🔴" if p == 1 else "Benign 🟢" for p in predictions]
                st.subheader("Results:")
                st.write(df[["Prediction"]])
            else:
                st.error("❌ CSV is missing some required columns.")
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
