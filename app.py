import streamlit as st
import numpy as np
from joblib import load
from apps.medical.diabetes_app import run as run_diabetes_app
from apps.house_price.house_price_app import run as run_house_price_app
from apps.chatbot.chatbot_app import run as run_chatbot_app
from apps.plant_disease.plant_disease_app import run as run_plant_disease_app

st.set_page_config(
    page_title="AI Portfolio",
    layout="wide"
)

st.title("Artificial Intelligence Portfolio")
st.write("Select a project from the menu to explore live AI demos.")

st.sidebar.title("Project Menu")

option = st.sidebar.selectbox(
    "Choose a Project",
    (
        "Chatbot",        
        "Medical Prediction",
        "House Price Forecasting",
        "Plant Disease Prediction", 
        "Movie Recommendation",
        "Best Seller Inventory Prediction",
        "Language Translation",
        "Smart Email Assistant (GenAI)"
    )
)

# -------------------------
# MEDICAL PREDICTION
# -------------------------
if option == "Medical Prediction":
    run_diabetes_app()

# -------------------------
# HOUSE VALUE PREDICTION
# -------------------------
elif option == "House Price Forecasting":
    st.header("Predict House Value")
    run_house_price_app()

# -------------------------
# CHATBOT
# -------------------------
elif option == "Chatbot":
    st.header("AI Longevity Chatbot")
    run_chatbot_app()

# -------------------------
# PLANT DISEASE PREDICTION
# -------------------------
elif option == "Plant Disease Prediction":
    st.header("Plant Disease Prediction")
    run_plant_disease_app()

# -------------------------
# PLACEHOLDERS
# -------------------------
elif option == "Movie Recommendation":
    st.header("Movie Recommendation System")
    st.info("Recommender system demo coming next.")

elif option == "Best Seller Inventory Prediction":
    st.header("Inventory Prediction")
    st.info("Forecasting demo coming next.")

elif option == "Language Translation":
    st.header("Language Translation")
    st.info("Translation demo coming next.")

elif option == "Smart Email Assistant (GenAI)":
    st.header("Smart Email Assistant")
    st.info("Generative AI demo coming next.")