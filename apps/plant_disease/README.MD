# 🌱 Plant Disease Detection Model (Computer Vision)

## Overview
This project is a computer vision–based plant disease detection system developed as part of my AI portfolio for employment. The goal is to demonstrate practical skills in deep learning, image preprocessing, model evaluation, and user-focused deployment using Streamlit.

The application allows users to upload an image of a plant leaf, optionally crop the diseased region, and receive a predicted plant health status along with treatment recommendations and confidence scores.

---

## Project Objectives
- Build a reliable image classification model for plant disease detection
- Compare a simple CNN with a ResNet-based architecture
- Improve generalization to real-world user images
- Communicate prediction confidence clearly and responsibly
- Deploy the model in an interactive web application

---

## Model Architecture
The final model uses **ResNet-18**, selected over a simple CNN due to:
- Residual connections that reduce vanishing gradients
- Better feature reuse and generalization
- Improved robustness on unseen images

The model outputs probabilities for each class using softmax.

---

## Supported Classes
- Healthy
- Early Blight
- Late Blight
- Leaf Miner
- Magnesium Deficiency
- Nitrogen Deficiency
- Potassium Deficiency
- Spotted Wilt Virus

---

## User Interface
The Streamlit app includes:
- Image upload
- Optional interactive cropping
- Prediction button (no auto-inference)
- Confidence bar chart showing class probabilities
- Clear explanation when a leaf is classified as Healthy based on a confidence threshold

A confidence threshold is used to avoid misleading users when predictions are uncertain.

---

## Ethical Considerations
This project does **not** provide medical or agricultural guarantees. Confidence scores are displayed to prevent over-trust in predictions, and a “Healthy” classification may occur when no disease exceeds the confidence threshold. This design prioritizes responsible AI use and transparency.

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Pillow)
- NumPy
- Matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
streamlit run apps/plant_disease/plant_disease_app.py
