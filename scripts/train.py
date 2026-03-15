import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from scripts.filter_dataset import filter_by_gsm_range
from scripts.dataset import GSMDataset
from scripts.model import GSMNet

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "gsm_ranges.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def train_for_cloth_type(cloth_type, epochs=5, batch_size=16):
    # Load GSM ranges
    with open(CONFIG_PATH, "r") as f:
        gsm_ranges = json.load(f)

    min_gsm = gsm_ranges[cloth_type]["min"]
    max_gsm = gsm_ranges[cloth_type]["max"]

    # Filter dataset by GSM range
    samples = filter_by_gsm_range(min_gsm, max_gsm)
    print(f"[{cloth_type}] Training on {len(samples)} images (GSM range {min_gsm}-{max_gsm})")

    if len(samples) == 0:
        raise RuntimeError("No images found in selected GSM range")

    # Dataset & DataLoader
    dataset = GSMDataset(samples, min_gsm, max_gsm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GSMNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, density in loader:
            images = images.to(device)
            density = density.to(device).float()


            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, density)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{cloth_type}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{cloth_type}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    cloth_types = [
    "heavyweight_cotton",
    "mediumweight_cotton",
    "lightweight_cotton",
    "linen",
    "heavyweight_denim"
]


    for cloth in cloth_types:
        print(f"\nTraining model for: {cloth}")
        train_for_cloth_type(cloth, epochs=3)

