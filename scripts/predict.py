import os
import json
import torch
from PIL import Image
from torchvision import transforms

from scripts.model import GSMNet
from scripts.gsm_utils import denormalize_gsm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "gsm_ranges.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def predict_gsm(image_path, cloth_type):
    """
    image_path: absolute or relative path to microscopic image
    cloth_type: string key from gsm_ranges.json
    """

    # Load GSM ranges
    with open(CONFIG_PATH, "r") as f:
        gsm_ranges = json.load(f)

    min_gsm = gsm_ranges[cloth_type]["min"]
    max_gsm = gsm_ranges[cloth_type]["max"]

    # Load model
    model_path = os.path.join(MODELS_DIR, f"{cloth_type}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for cloth type: {cloth_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GSMNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict density
    with torch.no_grad():
        density = model(image).item()

    # Convert density → GSM
    gsm = denormalize_gsm(density, min_gsm, max_gsm)

    return round(gsm, 2)
if __name__ == "__main__":
    test_image = "data/gsm/1-173/WIN_20260116_16_20_57_Pro.jpg"

    cloth_type = input("Enter cloth type (e.g. satin, cotton, wool): ").strip().lower()

    gsm = predict_gsm(test_image, cloth_type)
    print("Predicted GSM:", gsm)


