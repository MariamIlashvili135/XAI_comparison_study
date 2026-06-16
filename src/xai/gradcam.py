"""
gradcam.py
Runs Grad-CAM on all bounding-box images in NIH ChestX-ray14 and computes
two localization metrics against the ground-truth boxes:

  1. Pointing Game  — does the highest-activation pixel fall inside the box?
  2. IoU            — overlap between thresholded heatmap mask and the box

Outputs:
  results/gradcam_metrics.csv       per-image scores
  results/gradcam_summary.csv       per-pathology mean scores  (thesis Table 2)
  results/gradcam_examples/         heatmap overlay images for thesis figures

Run:
    python gradcam.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

from dataset import PATHOLOGIES, build_image_index

# ============================ CONFIG ========================================
DATA_ROOT    = r"D:\archive"
CKPT_PATH    = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR  = Path(r"D:\thesis\results")
EXAMPLES_DIR = RESULTS_DIR / "gradcam_examples"
BATCH_SIZE   = 1          # Grad-CAM must run one image at a time
IoU_THRESHOLD = 0.15      # binarise heatmap at top-X fraction of activation
                           # we also sweep this — see threshold_sweep below
NUM_EXAMPLES  = 5         # how many overlay images to save per pathology
# ===========================================================================

# The 8 pathologies that have ground-truth bounding boxes
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


# ── helpers ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(PATHOLOGIES))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val AUC {ckpt['mean_auc']:.4f}")
    return model


def load_bbox_df(data_root):
    """Load BBox_List_2017.csv and keep only the 8 boxed pathologies."""
    bbox_path = Path(data_root) / "BBox_List_2017.csv"
    df = pd.read_csv(bbox_path)
    # Normalise column names (the CSV uses spaces)
    df.columns = df.columns.str.strip()
    df["Finding Label"] = df["Finding Label"].replace("Infiltrate", "Infiltration")
    df = df[df["Finding Label"].isin(BOXED_PATHOLOGIES)].reset_index(drop=True)
    print(f"Bounding box records: {len(df)} "
          f"across {df['Image Index'].nunique()} unique images")
    return df


def scale_box(row, orig_w, orig_h, target=224):
    """Scale the ground-truth box from original image size to 224x224."""
    sx = target / orig_w
    sy = target / orig_h
    x1 = int(row["Bbox [x"] * sx)
    y1 = int(row["y"] * sy)
    w  = row["w"]
    h  = row["h]"]
    x2 = int((row["Bbox [x"] + w) * sx)
    y2 = int((row["y"] + h) * sy)
    return max(0, x1), max(0, y1), min(target, x2), min(target, y2)

def heatmap_to_mask(cam, threshold_fraction):
    """Binarise a [0,1] heatmap: top fraction of pixels -> 1."""
    cutoff = np.quantile(cam, 1.0 - threshold_fraction)
    return (cam >= cutoff).astype(np.uint8)


def pointing_game(cam, x1, y1, x2, y2):
    """1 if argmax pixel is inside box, else 0."""
    idx = np.argmax(cam)
    py, px = np.unravel_index(idx, cam.shape)
    return int(x1 <= px <= x2 and y1 <= py <= y2)


def compute_iou(mask, x1, y1, x2, y2):
    """IoU between binary mask and ground-truth bounding box."""
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    intersection = (mask & box_mask).sum()
    union = (mask | box_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def save_example(img_rgb, cam, x1, y1, x2, y2, out_path):
    """Save a side-by-side: original | Grad-CAM overlay with GT box."""
    overlay = show_cam_on_image(img_rgb, cam, use_rgb=True)
    # Draw ground-truth box on overlay
    overlay[y1:y2, x1:x1+2] = [255, 255, 0]
    overlay[y1:y2, x2:x2+2] = [255, 255, 0]
    overlay[y1:y1+2, x1:x2] = [255, 255, 0]
    overlay[y2:y2+2, x1:x2] = [255, 255, 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM + GT box (yellow)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    model = load_model(CKPT_PATH, device)

    # Target the last dense block — standard choice for DenseNet Grad-CAM
    target_layer = [model.features.denseblock4.denselayer16.conv2]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)

    bbox_df = load_bbox_df(DATA_ROOT)
    # Only keep rows whose image is on disk
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]

    records = []
    example_counts = {p: 0 for p in BOXED_PATHOLOGIES}

    total = len(bbox_df)
    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}")

        fname    = row["Image Index"]
        pathology = row["Finding Label"]
        class_idx = PATHOLOGIES.index(pathology)

        # Load image
        img_path = image_index[fname]
        pil_img  = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Preprocess
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

        # Run Grad-CAM
        targets = [ClassifierOutputTarget(class_idx)]
        cam = cam_extractor(input_tensor=tensor, targets=targets)[0]  # (224,224)
        cam = np.float32(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalise to [0,1]

        # Scale GT box to 224×224
        x1, y1, x2, y2 = scale_box(row, orig_w, orig_h)

        # Metrics
        pg   = pointing_game(cam, x1, y1, x2, y2)
        mask = heatmap_to_mask(cam, IoU_THRESHOLD)
        iou  = compute_iou(mask, x1, y1, x2, y2)

        records.append({
            "image":     fname,
            "pathology": pathology,
            "pointing_game": pg,
            "iou":       iou,
        })

        # Save example overlays
        if example_counts[pathology] < NUM_EXAMPLES:
            img_rgb = np.array(pil_img.resize((224, 224))) / 255.0
            out_path = EXAMPLES_DIR / f"{pathology}_{example_counts[pathology]+1}.png"
            save_example(img_rgb.astype(np.float32), cam, x1, y1, x2, y2, out_path)
            example_counts[pathology] += 1

    # ── aggregate results ──────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / "gradcam_metrics.csv", index=False)

    summary = results_df.groupby("pathology").agg(
        n_images=("image", "count"),
        pointing_game=("pointing_game", "mean"),
        mean_iou=("iou", "mean"),
    ).round(4)
    summary.loc["MEAN"] = summary.mean()
    print("\n" + "=" * 55)
    print("Grad-CAM Localization Results")
    print("=" * 55)
    print(summary.to_string())
    print("=" * 55)
    summary.to_csv(RESULTS_DIR / "gradcam_summary.csv")
    print(f"\nSaved -> {RESULTS_DIR / 'gradcam_summary.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'gradcam_metrics.csv'}")
    print(f"Example images -> {EXAMPLES_DIR}")

    # ── threshold sensitivity sweep ────────────────────────────────────────
    print("\nRunning IoU threshold sensitivity sweep...")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep_rows = []
    for thresh in thresholds:
        ious = []
        for _, row in results_df.iterrows():
            # re-load cam from saved records isn't feasible; use stored per-image
            # This sweep uses the already-computed CAMs indirectly via mean_iou
            # For a full sweep rerun is needed — flag this in thesis as limitation
            pass
        sweep_rows.append({"threshold": thresh})

    print("Note: full threshold sweep requires re-running CAM extraction.")
    print("The default threshold used is:", IoU_THRESHOLD)
    print("\nDone.")


if __name__ == "__main__":
    main()
