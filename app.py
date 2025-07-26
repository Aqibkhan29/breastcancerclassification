import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load CSS styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Load model
model = joblib.load("cancer_model.pkl")

st.markdown("<h1>🧬 Breast Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown("Predict whether a tumor is <b>Malignant 🔴</b> or <b>Benign 🟢</b> using the Breast Cancer Wisconsin dataset.", unsafe_allow_html=True)

option = st.radio("Select input method:", ("Manual Entry", "Upload CSV"))

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

if option == "Manual Entry":
    st.subheader("🔢 Enter Patient Data:")
    input_values = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            val = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, format="%.5f")
            input_values.append(val)

    if st.button("Predict"):
        input_data = np.array([input_values])
        prediction = model.predict(input_data)
        result = "🔴 Malignant" if prediction[0] == 1 else "🟢 Benign"
        st.success(f"Prediction Result: {result}")

else:
    uploaded_file = st.file_uploader("📄 Upload CSV with 30 features", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if all(feature in df.columns for feature in features):
                predictions = model.predict(df[features])
                df["Prediction"] = ["Malignant 🔴" if p == 1 else "Benign 🟢" for p in predictions]
                st.success("✅ Prediction completed.")
                st.write(df[["Prediction"]])
            else:
                st.error("❌ Some required columns are missing in the CSV.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
