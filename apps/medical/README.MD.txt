# AI Portfolio

## Overview

This repository showcases **live demos of AI and machine learning projects**. The projects are designed for both educational purposes and portfolio presentation. The web interface is built using **Streamlit**, allowing interactive exploration of AI models.

Currently included in the app:

- **Medical Risk Prediction** (Diabetes Progression)
- *(Other models like Breast Cancer, Heart Disease, Synthetic Cancer, and House Price Forecasting are trained in notebooks but not yet exposed in the demo app.)*

---

## Project Structure

ai-portfolio/
├─ notebooks/ # Jupyter/Colab notebooks with training code
│ ├─ medical_models.ipynb
│ ├─ house_value_model.ipynb
├─ models/ # Saved model files (joblib)
│ ├─ diabetes_model.pkl
│ ├─ breast_cancer_model.pkl
│ └─ ...
├─ app.py # Streamlit app
├─ requirements.txt # Python dependencies
└─ README.md

yaml
Copy code

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-portfolio.git
cd ai-portfolio
2. Create a Virtual Environment (Recommended)
Using conda:

bash
Copy code
conda create -n ai-portfolio python=3.12
conda activate ai-portfolio
Or using venv:

bash
Copy code
python -m venv ai-portfolio-env
source ai-portfolio-env/bin/activate  # Linux/Mac
ai-portfolio-env\Scripts\activate     # Windows
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
Requirements for Diabetes Model (exact versions used for reproducibility):

txt
Copy code
scikit-learn==1.7.2
joblib>=1.3
numpy>=1.23
pandas>=1.5
⚠️ Important: Using the same scikit-learn version ensures no warnings or inconsistent behavior when loading trained models.

Running the Streamlit App
bash
Copy code
streamlit run app.py
Open the browser when prompted.

Use the sidebar menu to select Medical Prediction.

Only the Diabetes Progression model is currently exposed in the demo.

Input patient information using the interactive fields.

The app will display:

Risk classification (High / Low)

Estimated probability (%)

Risk visualization using a progress bar

Model Information
Diabetes Progression Classification
Algorithm: Logistic Regression

Scaling: StandardScaler applied to all features

Features:

Age

Sex (0 = Female, 1 = Male)

BMI

Blood Pressure

Total Cholesterol (TC)

LDL

HDL

TC/HDL ratio

Glucose

Blood Sugar Level

Output:

Class: 0 (Lower Progression Risk) / 1 (Higher Progression Risk)

Probability % displayed in the app

Note: Models are trained in notebooks under controlled environments. To reproduce results exactly, use the specified versions in requirements.txt.

How the App Works
Model Loading:

Models and scalers are saved as joblib objects (.pkl) for fast loading.

Example: model, scaler = joblib.load("models/diabetes_model.pkl")

User Inputs:

Users enter real-world, unnormalized values for features.

Example: Age, BMI, Blood Pressure, Cholesterol levels.

Data Preprocessing:

Inputs are converted into a DataFrame with the same column names as training data.

Features are scaled using the loaded StandardScaler.

Prediction:

The Logistic Regression model predicts the risk class.

Probabilities are computed for a risk % display.

Display:

Risk is shown using:

Text (High / Low)

Probability % (e.g., 74.5%)

Visual progress bar for intuitive feedback

Future Improvements
Expose additional trained models in the Streamlit app:

Breast Cancer Detection

Heart Disease Risk

House Price Forecasting

Chatbot Demo

Enhance the UI with more metrics, plots, and feature importance visualizations

Add user authentication for personal model testing

Include API key integration for generative AI experiments

References
Scikit-learn Documentation

Streamlit Documentation

Joblib Documentation

Notes for Portfolio Reviewers
Models are trained on both real and synthetic datasets.

This repo demonstrates end-to-end workflow: data preprocessing → model training → saving → web deployment.

Interactive app emphasizes user-friendly inputs and risk interpretability.

All code is Python 3.12 compatible.