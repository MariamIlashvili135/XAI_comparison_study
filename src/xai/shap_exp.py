"""
shap_exp.py  (v4 — Captum GradientShap, DenseNet-121 compatible)

Uses Captum's GradientShap instead of the shap library directly.
GradientShap is mathematically equivalent to SHAP's GradientExplainer
and is fully compatible with DenseNet-121's architecture.

In your thesis this is correctly cited as "SHAP (GradientSHAP variant)"
which is standard practice in XAI literature on medical imaging.

Outputs:
  results/shap_metrics.csv
  results/shap_summary.csv      (thesis Table 4)
  results/shap_examples/

Run (from D:\\thesis\\src\\xai):
    python shap_exp.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import GradientShap
from PIL import Image
from torchvision import models, transforms

from dataset import PATHOLOGIES, build_image_index

# ============================ CONFIG ========================================
DATA_ROOT       = r"D:\archive"
CKPT_PATH       = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR     = Path(r"D:\thesis\results")
EXAMPLES_DIR    = RESULTS_DIR / "shap_examples"
BACKGROUND_SIZE = 20
IoU_THRESHOLD   = 0.15
NUM_EXAMPLES    = 5
N_SAMPLES       = 10   # GradientShap samples per image — 10 is standard
# ===========================================================================

BOXED_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(ckpt_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(PATHOLOGIES))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val AUC {ckpt['mean_auc']:.4f}")
    return model


def load_bbox_df(data_root):
    df = pd.read_csv(Path(data_root) / "BBox_List_2017.csv")
    df.columns = df.columns.str.strip()
    df["Finding Label"] = df["Finding Label"].replace("Infiltrate", "Infiltration")
    df = df[df["Finding Label"].isin(BOXED_PATHOLOGIES)].reset_index(drop=True)
    print(f"Bounding box records: {len(df)} across {df['Image Index'].nunique()} images")
    return df


def build_background(bbox_df, image_index, device, n=20):
    """Sample n images as GradientShap baseline distribution."""
    sample = bbox_df.sample(n=min(n, len(bbox_df)), random_state=42)
    tensors = []
    for _, row in sample.iterrows():
        pil = Image.open(image_index[row["Image Index"]]).convert("RGB")
        tensors.append(TRANSFORM(pil))
    bg = torch.stack(tensors).to(device)
    print(f"  background tensor shape: {bg.shape}")
    return bg


def heatmap_to_mask(attr_map, threshold_fraction=0.15):
    cutoff = np.quantile(attr_map, 1.0 - threshold_fraction)
    return (attr_map >= cutoff).astype(np.uint8)


def pointing_game(attr_map, x1, y1, x2, y2):
    idx = np.argmax(attr_map)
    py, px = np.unravel_index(idx, attr_map.shape)
    return int(x1 <= px <= x2 and y1 <= py <= y2)


def compute_iou(mask, x1, y1, x2, y2):
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    intersection = (mask & box_mask).sum()
    union = (mask | box_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def save_example(img_rgb, attr_map, x1, y1, x2, y2, out_path):
    a = attr_map.copy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    # Blend SHAP heatmap on top of original X-ray so anatomy stays visible
    img_float = img_rgb.astype(np.float32) / 255.0
    heatmap = plt.cm.RdBu_r(a)[:, :, :3]
    overlay = img_float * 0.5 + heatmap * 0.5
    overlay = np.clip(overlay, 0, 1)
    overlay[y1:y2, x1:x1+2] = [1, 1, 0]
    overlay[y1:y2, x2:x2+2] = [1, 1, 0]
    overlay[y1:y1+2, x1:x2] = [1, 1, 0]
    overlay[y2:y2+2, x1:x2] = [1, 1, 0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("GradientSHAP + GT box (yellow)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    model = load_model(CKPT_PATH, device)

    bbox_df = load_bbox_df(DATA_ROOT)
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]

    print(f"Building background distribution ({BACKGROUND_SIZE} images)...")
    background = build_background(bbox_df, image_index, device, BACKGROUND_SIZE)

    # Captum GradientShap — wraps model to accept single input tensor
    print("Initialising GradientShap explainer...")
    explainer = GradientShap(model)
    print("GradientShap ready. Starting inference loop...")

    records = []
    example_counts = {p: 0 for p in BOXED_PATHOLOGIES}
    total = len(bbox_df)
    failed = 0

    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}  (failed so far: {failed})")

        fname     = row["Image Index"]
        pathology = row["Finding Label"]
        class_idx = PATHOLOGIES.index(pathology)

        pil_img = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size

        # requires_grad=True is needed for Captum
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device).requires_grad_(True)

        # Scale GT box to 224x224
        sx = 224 / orig_w
        sy = 224 / orig_h
        x1 = max(0,   int(row["Bbox [x"] * sx))
        y1 = max(0,   int(row["y"] * sy))
        h  = row["h]"]
        x2 = min(224, int((row["Bbox [x"] + row["w"]) * sx))
        y2 = min(224, int((row["y"] + h) * sy))

        try:
            # GradientShap: sample baselines from background,
            # compute attributions for the target class
            attr = explainer.attribute(
                tensor,
                baselines=background,
                target=class_idx,
                n_samples=N_SAMPLES,
                stdevs=0.09,
            )
            # attr shape: (1, 3, 224, 224)
            attr_map = attr[0].detach().cpu().numpy()   # (3, 224, 224)
            attr_map = np.abs(attr_map).mean(axis=0)    # (224, 224)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  WARNING: failed for {fname}: {e}")
            records.append({"image": fname, "pathology": pathology,
                            "pointing_game": 0, "iou": 0.0})
            continue

        pg   = pointing_game(attr_map, x1, y1, x2, y2)
        mask = heatmap_to_mask(attr_map, IoU_THRESHOLD)
        iou  = compute_iou(mask, x1, y1, x2, y2)

        records.append({"image": fname, "pathology": pathology,
                        "pointing_game": pg, "iou": iou})

        if example_counts[pathology] < NUM_EXAMPLES:
            img_rgb = np.array(pil_img.resize((224, 224)))
            out_path = EXAMPLES_DIR / f"{pathology}_{example_counts[pathology]+1}.png"
            try:
                save_example(img_rgb, attr_map, x1, y1, x2, y2, out_path)
                example_counts[pathology] += 1
            except Exception:
                pass

    print(f"\nTotal failed: {failed}/{total}")
    if failed == total:
        print("ERROR: All images failed — paste this output for further help.")
        return

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / "shap_metrics.csv", index=False)

    summary = results_df.groupby("pathology").agg(
        n_images=("image", "count"),
        pointing_game=("pointing_game", "mean"),
        mean_iou=("iou", "mean"),
    ).round(4)
    summary.loc["MEAN"] = summary.mean()

    print("\n" + "=" * 55)
    print("SHAP (GradientSHAP) Localization Results")
    print("=" * 55)
    print(summary.to_string())
    print("=" * 55)
    summary.to_csv(RESULTS_DIR / "shap_summary.csv")
    print(f"\nSaved -> {RESULTS_DIR / 'shap_summary.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'shap_metrics.csv'}")
    print(f"Example images -> {RESULTS_DIR / 'shap_examples'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
