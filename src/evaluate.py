"""
evaluate.py
Evaluates the trained DenseNet-121 checkpoint on the official held-out test set.

Produces:
  - Printed per-class AUC table in the terminal
  - results/test_auc_results.csv  (paste into your thesis)
  - results/auc_chart.png         (bar chart for your thesis figures)

Run:
    python evaluate.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import models

from dataset import (
    ChestXray14, PATHOLOGIES, build_image_index,
    get_transforms, load_dataframe, read_list,
)

# ============================ CONFIG — EDIT IF NEEDED =======================
DATA_ROOT   = r"D:\archive"
CKPT_PATH   = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR = Path(r"D:\thesis\results")
BATCH_SIZE  = 16
NUM_WORKERS = 6
# ===========================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(ckpt_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(PATHOLOGIES))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC {ckpt['mean_auc']:.4f})")
    return model


def run_inference(model, loader, device):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda"):
                out = model(imgs)
            all_probs.append(torch.sigmoid(out).float().cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_aucs(y_true, y_prob):
    aucs = []
    for c in range(len(PATHOLOGIES)):
        col = y_true[:, c]
        if 0 < col.sum() < len(col):
            aucs.append(roc_auc_score(col, y_prob[:, c]))
        else:
            aucs.append(float("nan"))
    return aucs


def print_table(aucs):
    print("\n" + "=" * 40)
    print(f"{'Pathology':<22} {'AUC':>6}")
    print("=" * 40)
    for name, auc in zip(PATHOLOGIES, aucs):
        marker = "  <-- below target" if not np.isnan(auc) and auc < 0.80 else ""
        print(f"{name:<22} {auc:>6.3f}{marker}")
    print("-" * 40)
    mean = float(np.nanmean(aucs))
    print(f"{'Mean AUC':<22} {mean:>6.3f}")
    print("=" * 40 + "\n")
    return mean


def save_csv(aucs, mean_auc):
    rows = [{"Pathology": n, "AUC": round(a, 4)}
            for n, a in zip(PATHOLOGIES, aucs)]
    rows.append({"Pathology": "Mean", "AUC": round(mean_auc, 4)})
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "test_auc_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved CSV -> {out}")


def save_chart(aucs, mean_auc):
    valid = [(n, a) for n, a in zip(PATHOLOGIES, aucs) if not np.isnan(a)]
    names, values = zip(*sorted(valid, key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if v >= 0.80 else "#e74c3c" for v in values]
    bars = ax.barh(names, values, color=colors)
    ax.axvline(mean_auc, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axvline(0.80, color="orange", linestyle=":", linewidth=1.5,
               label="Target AUC = 0.80")
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_title("DenseNet-121 Test Set AUC — NIH ChestX-ray14", fontsize=13)
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "auc_chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved chart -> {out}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    df = load_dataframe(Path(DATA_ROOT) / "Data_Entry_2017.csv")
    df = df[df["Image Index"].isin(image_index.keys())]

    test_names = read_list(Path(DATA_ROOT) / "test_list.txt")
    test_df = df[df["Image Index"].isin(test_names)]
    print(f"Test set size: {len(test_df)} images")

    test_ds = ChestXray14(test_df, image_index, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = load_model(CKPT_PATH, device)

    print("Running inference on test set (this takes a few minutes)...")
    y_true, y_prob = run_inference(model, test_loader, device)

    aucs = compute_aucs(y_true, y_prob)
    mean_auc = print_table(aucs)
    save_csv(aucs, mean_auc)
    save_chart(aucs, mean_auc)

    print("Done. Files saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
