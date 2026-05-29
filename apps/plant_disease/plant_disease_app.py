# apps/plant_disease/plant_disease_app.py

import sys
from pathlib import Path
import warnings

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd

# -------------------------
# Silence Streamlit warnings
# -------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------
# Add project root to sys.path
# -------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # ai-portfolio
sys.path.append(str(ROOT_DIR))

# -------------------------
# Import inference components
# -------------------------
from apps.plant_disease.inference import model, transform, CLASS_NAMES, device

# -------------------------
# Optional cropper
# -------------------------
try:
    from streamlit_cropper import st_cropper
except ImportError:
    st_cropper = None
    st.warning("⚠️ Install cropper: pip install streamlit-cropper")

# -------------------------
# Treatment recommendations
# -------------------------
RECOMMENDATIONS = {
    "Healthy": "No action needed. Keep monitoring. No disease exceeded the confidence threshold, so the leaf is classified as Healthy.",
    "Early_blight": "Remove infected leaves and apply a fungicide containing chlorothalonil or copper.",
    "Late_blight": "Remove infected leaves and apply a fungicide containing mancozeb or copper.",
    "Leaf Miner": "Apply neem oil or insecticides targeting leaf miner larvae. Remove affected leaves.",
    "Magnesium Deficiency": "Apply magnesium-containing fertilizers or foliar sprays.",
    "Nitrogen Deficiency": "Apply nitrogen-rich fertilizers and monitor soil nutrient levels.",
    "Pottassium Deficiency": "Apply potassium-containing fertilizers and maintain soil pH.",
    "Spotted Wilt Virus": "Remove infected plants immediately and control thrips with insecticides or yellow sticky traps."
}

# -------------------------
# Confidence threshold
# -------------------------
HEALTHY_THRESHOLD = 0.65  # conservative default

# -------------------------
# Main app
# -------------------------
def run():
    st.title("🌱 Plant Disease Prediction Demo")

    st.write(
        "Upload a leaf image, optionally crop the affected area, "
        "and click **Predict Disease** to see the model output and confidence."
    )

    # -------- Upload image --------
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    cropped_image = image

    # -------- Optional crop --------
    if st_cropper is not None:
        if st.checkbox("Optional.  ✂️ Crop image before prediction"):
            cropped_image = st_cropper(
                image,
                realtime_update=False,
                box_color="#FF0000",
                aspect_ratio=None
            )

    st.image(cropped_image, caption="Image used for prediction", width=400)

    # -------- Predict button --------
    if st.button("Predict Disease"):
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu()

        max_prob, pred_idx = torch.max(probs, dim=0)

        if max_prob.item() < HEALTHY_THRESHOLD:
            pred_class = "Healthy"
        else:
            pred_class = CLASS_NAMES[pred_idx.item()]

        treatment = RECOMMENDATIONS.get(
            pred_class,
            "No recommendation available."
        )

        # -------- Results --------
        st.success(f"✅ Predicted Status: **{pred_class}**")
        st.info(f"💡 Recommended Action: {treatment}")

        # -------- Confidence bar chart --------
        st.subheader("🔍 Prediction Confidence")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs.numpy()
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Class"))

        # -------- Optional numeric display --------
        st.caption(
            f"Top prediction confidence: **{max_prob.item():.2%}** "
            "(probabilities are model estimates, not guarantees)"
        )


# -------------------------
# Run standalone
# -------------------------
if __name__ == "__main__":
    run()


