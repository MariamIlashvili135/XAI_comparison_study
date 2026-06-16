"""
dataset.py
PyTorch Dataset and data helpers for the NIH ChestX-ray14 dataset.

This module handles:
  - Encoding the 14 pathology labels from Data_Entry_2017.csv into a multi-hot vector
  - Building a {filename -> full path} index so it works no matter how your image
    folders are laid out on disk (images_001 ... images_012, or flat, etc.)
  - Train/val/test transforms (ImageNet normalization, since DenseNet is pretrained)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# The 14 official NIH ChestX-ray14 pathology labels.
# NOTE: only 8 of these have ground-truth bounding boxes (in BBox_List_2017.csv):
# Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
# Pneumothorax. Your IoU / pointing-game evaluation can only run on those 8.
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_index(data_root):
    """Recursively scan data_root for every .png and map filename -> full path.

    This is run once at startup so the Dataset doesn't care about folder layout.
    """
    data_root = Path(data_root)
    index = {}
    for p in data_root.rglob("*.png"):
        index[p.name] = p
    return index


def load_dataframe(csv_path):
    """Read Data_Entry_2017.csv and add one binary column per pathology."""
    df = pd.read_csv(csv_path)
    for path in PATHOLOGIES:
        df[path] = df["Finding Labels"].apply(
            lambda s, p=path: 1 if p in str(s).split("|") else 0
        )
    return df


def read_list(path):
    """Read an official split file (train_val_list.txt / test_list.txt)."""
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def get_transforms(train=True):
    """Image transforms. Light augmentation for training only.

    Note: we deliberately do NOT horizontal-flip chest X-rays, because left/right
    asymmetry (e.g. cardiac silhouette, situs) is clinically meaningful.
    """
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomRotation(7),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXray14(Dataset):
    """Returns (image_tensor[3,224,224], label_tensor[14])."""

    def __init__(self, df, image_index, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.image_index[row["Image Index"]]
        img = Image.open(path).convert("RGB")  # grayscale -> 3 channels
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(
            row[PATHOLOGIES].to_numpy(dtype="float32")
        )
        return img, label
