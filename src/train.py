"""
train.py
Trains a DenseNet-121 multi-label classifier on NIH ChestX-ray14.

Tuned for an RTX 3050 Ti Laptop GPU (4 GB VRAM):
  - mixed-precision (AMP) training to roughly halve memory and speed things up
  - modest batch size (see CONFIG; drop to 8 if you hit "CUDA out of memory")

Outputs the best checkpoint (by mean validation AUC) to checkpoints/.

Run:  (with your venv activated)
    python train.py
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models

from dataset import (
    ChestXray14, PATHOLOGIES, build_image_index,
    get_transforms, load_dataframe, read_list,
)

# ============================ CONFIG — EDIT THIS ============================
# Point this at the folder that contains Data_Entry_2017.csv, the split .txt
# files, and the image folders. Use a raw string (r"...") on Windows.
DATA_ROOT = r"D:\archive"          # <-- EDIT ME

BATCH_SIZE = 16     # 4GB VRAM: start at 16. If you see "CUDA out of memory", set 8.
NUM_WORKERS = 6     # you have 20 logical cores; if Windows acts up, set to 0 or 2.
EPOCHS = 20
LR = 1e-4
VAL_FRACTION = 0.1  # fraction of train_val patients held out for validation
SEED = 42
# ===========================================================================

DATA_ROOT = Path(DATA_ROOT)
CSV_PATH = DATA_ROOT / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = DATA_ROOT / "train_val_list.txt"
TEST_LIST = DATA_ROOT / "test_list.txt"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_split(df, val_fraction, seed):
    """Split by Patient ID (NOT by image) to avoid patient leakage."""
    patients = df["Patient ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = int(len(patients) * val_fraction)
    val_patients = set(patients[:n_val])
    val_df = df[df["Patient ID"].isin(val_patients)]
    train_df = df[~df["Patient ID"].isin(val_patients)]
    return train_df, val_df


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda"):
                out = model(imgs)
            all_probs.append(torch.sigmoid(out).float().cpu().numpy())
            all_labels.append(labels.numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    aucs = []
    for c in range(len(PATHOLOGIES)):
        col = y_true[:, c]
        if 0 < col.sum() < len(col):
            aucs.append(roc_auc_score(col, y_prob[:, c]))
        else:
            aucs.append(float("nan"))  # class absent in val -> undefined AUC
    return aucs


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: CUDA not available — training will be extremely slow.")
        print("Re-check your PyTorch install (see the install instructions).")

    print("Indexing image files (one-time scan, may take ~1 min)...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    df = load_dataframe(CSV_PATH)
    # Keep only rows whose image we actually have on disk (robust to partial downloads)
    df = df[df["Image Index"].isin(image_index.keys())]

    train_val_names = read_list(TRAIN_VAL_LIST)
    test_names = read_list(TEST_LIST)
    trainval_df = df[df["Image Index"].isin(train_val_names)]
    test_df = df[df["Image Index"].isin(test_names)]  # held out — used later, in evaluate.py
    train_df, val_df = patient_split(trainval_df, VAL_FRACTION, SEED)

    print(f"Split sizes — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    train_ds = ChestXray14(train_df, image_index, get_transforms(train=True))
    val_ds = ChestXray14(val_df, image_index, get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Class imbalance: positive weight = (#negatives / #positives) per class.
    pos = train_df[PATHOLOGIES].sum().to_numpy()
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32).to(device)

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(PATHOLOGIES))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # multi-label -> BCE, not CE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=1
    )
    scaler = GradScaler("cuda")

    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch}  step {i+1}/{len(train_loader)}  "
                      f"loss {running/100:.4f}")
                running = 0.0

        aucs = evaluate(model, val_loader, device)
        mean_auc = float(np.nanmean(aucs))
        print(f"\nEpoch {epoch}  mean val AUC = {mean_auc:.4f}")
        for name, a in zip(PATHOLOGIES, aucs):
            print(f"    {name:20s} {a:.3f}")
        scheduler.step(mean_auc)

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mean_auc": mean_auc},
                CKPT_DIR / "densenet121_best.pt",
            )
            print(f"  -> saved new best checkpoint ({mean_auc:.4f})\n")

    print("Training complete. Best mean val AUC:", round(best_auc, 4))


if __name__ == "__main__":
    main()
