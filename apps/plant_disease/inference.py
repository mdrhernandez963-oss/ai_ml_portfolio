# apps/plant_disease/inference.py

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------
# Device (CPU/GPU)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Class names (must match training)
# -------------------------
CLASS_NAMES = [
    "Healthy",
    "Early_blight",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]

# -------------------------
# Image preprocessing (match ResNet18 training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Load ResNet18 model
# -------------------------
def load_model(weights_path: Path) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model weights not found at:\n{weights_path}\n"
            "✔ Place your ResNet18 .pth file inside 'ai-portfolio/models/'"
        )

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=False)  # set pretrained=False since weights loaded
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))  # adjust final layer

    # Load trained weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Paths
# -------------------------
_THIS_DIR = Path(__file__).resolve().parent          # apps/plant_disease
_PROJECT_ROOT = _THIS_DIR.parent.parent             # ai-portfolio
_MODEL_PATH = _PROJECT_ROOT / "models" / "plant_disease_resnet18.pth"

# -------------------------
# Global model instance (loaded once)
# -------------------------
model = load_model(_MODEL_PATH)
