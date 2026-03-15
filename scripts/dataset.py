import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scripts.gsm_utils import normalize_gsm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


class GSMDataset(Dataset):
    def __init__(self, samples, min_gsm, max_gsm, transform=None):
        """
        samples: list of (image_path, gsm)
        """
        self.samples = samples
        self.min_gsm = min_gsm
        self.max_gsm = max_gsm

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_image_path, gsm = self.samples[idx]
        image_path = os.path.join(DATA_DIR, rel_image_path)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        density = normalize_gsm(gsm, self.min_gsm, self.max_gsm)

        return image, density
