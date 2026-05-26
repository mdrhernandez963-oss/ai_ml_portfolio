# apps/medical/diabetes_app.py

import streamlit as st
import pandas as pd
from joblib import load


@st.cache_resource
def load_diabetes_model():
    model, scaler = load("models/diabetes_model.pkl")
    return model, scaler


def run():
    st.header("🩺 Medical Risk Prediction (Diabetes Progression)")

    model, scaler = load_diabetes_model()

    st.subheader("Enter Patient Information")

    age = st.number_input("Age (19–95 years)", 19, 95, 45)
    sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1, 0, step=1)
    bmi = st.number_input("BMI (18–40 kg/m²)", 18.0, 40.0, 25.0)
    bp = st.number_input("Blood Pressure (80–180 mmHg)", 80.0, 180.0, 120.0)
    s1 = st.number_input("Total Cholesterol (120–240 mg/dL)", 120.0, 240.0, 200.0)
    s2 = st.number_input("LDL (70–160 mg/dL)", 70.0, 160.0, 130.0)
    s3 = st.number_input("HDL (40–70 mg/dL)", 40.0, 70.0, 50.0)
    s4 = st.number_input("TC/HDL Ratio (2–6)", 2.0, 6.0, 4.0)
    s5 = st.number_input("Glucose (70–130 mg/dL)", 70.0, 130.0, 100.0)
    s6 = st.number_input("Blood Sugar Level (80–150 mg/dL)", 80.0, 150.0, 110.0)

    if st.button("Predict Diabetes Risk"):
        feature_names = [
            "age", "sex", "bmi", "bp",
            "s1", "s2", "s3", "s4",
            "s5", "s6"
        ]

        input_df = pd.DataFrame([[ 
            age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
        ]], columns=feature_names)

        scaled_df = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_names
        )

        prediction = model.predict(scaled_df)[0]
        probability = model.predict_proba(scaled_df)[0][1] * 100

        if prediction == 1:
            st.error("""⚠️ High risk of diabetes progression detected.

        Clinical Interpretation:
        → Increased risk of chronic complications
        → Retinal screening recommended
        → Foot examination recommended
        """) 
        else:
            st.success("✅ Low risk of diabetes progression.")

        st.metric("Estimated Risk Probability", f"{probability:.1f} %")
        st.progress(min(int(probability), 100))
