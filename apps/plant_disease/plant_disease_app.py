# apps/plant_disease/plant_disease_app.py

import sys
from pathlib import Path
import warnings

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# -------------------------
# Silence warnings
# -------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------
# Add project root to sys.path
# -------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# -------------------------
# Import inference components
# -------------------------
from apps.plant_disease.inference import model, transform, CLASS_NAMES, device

# -------------------------
# Treatment recommendations
# -------------------------
RECOMMENDATIONS = {
    "Healthy": "No action needed. Keep monitoring.",
    "Early_blight": "Remove infected leaves and apply chlorothalonil or copper fungicide.",
    "Late_blight": "Remove infected leaves and apply mancozeb or copper fungicide.",
    "Leaf Miner": "Use neem oil or insecticides and remove affected leaves.",
    "Magnesium Deficiency": "Apply magnesium fertilizer or foliar spray.",
    "Nitrogen Deficiency": "Apply nitrogen-rich fertilizer.",
    "Pottassium Deficiency": "Apply potassium fertilizer and adjust soil pH.",
    "Spotted Wilt Virus": "Remove infected plants and control thrips."
}

HEALTHY_THRESHOLD = 0.65


# -------------------------
# Main app
# -------------------------
def run():

    st.title("🌱 Plant Disease Prediction Demo")

    st.write(
        "Upload a leaf image and click **Predict Disease** to get results."
    )

    # -------------------------
    # Upload image
    # -------------------------
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    # -------------------------
    # Load image safely
    # -------------------------
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("Predict Disease"):

        # -------------------------
        # SAFE IMAGE HANDLING
        # -------------------------
        try:
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            st.error(f"Image preprocessing failed: {e}")
            st.stop()

        # -------------------------
        # Model inference
        # -------------------------
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu()

        max_prob, pred_idx = torch.max(probs, dim=0)

        # -------------------------
        # Decision logic
        # -------------------------
        if max_prob.item() < HEALTHY_THRESHOLD:
            pred_class = "Healthy"
        else:
            pred_class = CLASS_NAMES[pred_idx.item()]

        treatment = RECOMMENDATIONS.get(
            pred_class,
            "No recommendation available."
        )

        # -------------------------
        # Results
        # -------------------------
        st.success(f"✅ Predicted Disease: **{pred_class}**")
        st.info(f"💡 Recommendation: {treatment}")

        # -------------------------
        # Confidence chart
        # -------------------------
        st.subheader("🔍 Prediction Confidence")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs.numpy()
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Class"))

        st.caption(
            f"Top confidence: **{max_prob.item():.2%}**"
        )


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    run()
