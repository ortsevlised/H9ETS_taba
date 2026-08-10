import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"


def read_codecarbon_energy_kwh(csv_path):
    """CodeCarbon's tracker.stop() only returns emissions (kg CO2e); the
    energy actually consumed (kWh) is written to the tracker's output CSV
    but not returned directly. Read it back from the last row."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["energy_consumed"])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, split, transform):
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} in {manifest_csv}")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = IMAGES_DIR / row["archive_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(float(row["label"]))
        return img, label, row["source"]


def build_model(name):
    """Binary classification head (single logit, use with BCEWithLogitsLoss)."""
    if name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_features, 1)
        return m
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_features, 1)
        return m
    if name == "xception":
        import timm
        m = timm.create_model("xception", pretrained=True, num_classes=1)
        return m
    raise ValueError(f"unknown model name {name!r}")


MODEL_NAMES = ["mobilenet_v3_large", "efficientnet_b0", "xception"]
