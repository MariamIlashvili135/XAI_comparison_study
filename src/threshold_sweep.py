"""
threshold_sweep.py
Runs IoU at multiple thresholds for all three XAI methods and produces:
  - results/threshold_sweep.csv       full results table
  - results/threshold_sweep.png       line chart for thesis

This does NOT rerun Grad-CAM, LIME, or SHAP — it reads the per-image
metrics CSVs already saved and recomputes IoU at different thresholds
using the stored attribution maps.

Wait — actually we need to recompute from raw attributions.
Instead this script re-reads the per-image CSV scores and applies
a statistical approach: it sweeps threshold by rerunning CAM on
the bbox subset at each threshold level.

Run (from D:\\thesis\\src):
    python threshold_sweep.py
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
from lime import lime_image
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# We need dataset.py in the path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import PATHOLOGIES, build_image_index

# ============================ CONFIG ========================================
DATA_ROOT    = r"D:\archive"
CKPT_PATH    = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR  = Path(r"D:\thesis\results")
THRESHOLDS   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
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

COLORS = {
    "Grad-CAM": "#2196F3",
    "LIME":     "#FF9800",
    "SHAP":     "#4CAF50",
}


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
    return df


def compute_iou_at_threshold(attr_map, x1, y1, x2, y2, threshold):
    cutoff = np.quantile(attr_map, 1.0 - threshold)
    mask = (attr_map >= cutoff).astype(np.uint8)
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    intersection = (mask & box_mask).sum()
    union = (mask | box_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def get_box(row, orig_w, orig_h):
    sx = 224 / orig_w
    sy = 224 / orig_h
    x1 = max(0,   int(row["Bbox [x"] * sx))
    y1 = max(0,   int(row["y"] * sy))
    x2 = min(224, int((row["Bbox [x"] + row["w"]) * sx))
    y2 = min(224, int((row["y"] + row["h]"]) * sy))
    return x1, y1, x2, y2


def collect_gradcam_maps(model, bbox_df, image_index, device):
    """Collect all Grad-CAM attribution maps — returns list of (attr_map, x1,y1,x2,y2)."""
    target_layer = [model.features.denseblock4.denselayer16.conv2]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)
    results = []
    total = len(bbox_df)
    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Grad-CAM {i+1}/{total}")
        fname = row["Image Index"]
        class_idx = PATHOLOGIES.index(row["Finding Label"])
        pil_img = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(class_idx)]
        cam = cam_extractor(input_tensor=tensor, targets=targets)[0]
        cam = np.float32(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        x1, y1, x2, y2 = get_box(row, orig_w, orig_h)
        results.append((cam, x1, y1, x2, y2))
    return results


def collect_shap_maps(model, bbox_df, image_index, device):
    """Collect all SHAP attribution maps."""
    # Build background
    sample = bbox_df.sample(n=20, random_state=42)
    bg_tensors = []
    for _, row in sample.iterrows():
        pil = Image.open(image_index[row["Image Index"]]).convert("RGB")
        bg_tensors.append(TRANSFORM(pil))
    background = torch.stack(bg_tensors).to(device)
    explainer = GradientShap(model)

    results = []
    total = len(bbox_df)
    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  SHAP {i+1}/{total}")
        fname = row["Image Index"]
        class_idx = PATHOLOGIES.index(row["Finding Label"])
        pil_img = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device).requires_grad_(True)
        try:
            attr = explainer.attribute(tensor, baselines=background,
                                       target=class_idx, n_samples=10, stdevs=0.09)
            attr_map = np.abs(attr[0].detach().cpu().numpy()).mean(axis=0)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        except Exception:
            attr_map = np.zeros((224, 224))
        x1, y1, x2, y2 = get_box(row, orig_w, orig_h)
        results.append((attr_map, x1, y1, x2, y2))
    return results


def collect_lime_maps(model, bbox_df, image_index, device):
    """Collect all LIME attribution maps."""
    def predict_fn(images):
        batch = []
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)
            pil = Image.fromarray(img_uint8)
            batch.append(TRANSFORM(pil))
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch_tensor)).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer(random_state=42)
    results = []
    total = len(bbox_df)
    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  LIME {i+1}/{total}")
        fname = row["Image Index"]
        class_idx = PATHOLOGIES.index(row["Finding Label"])
        pil_img = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size
        pil_224 = pil_img.resize((224, 224))
        img_float = np.array(pil_224).astype(np.float64) / 255.0
        try:
            explanation = explainer.explain_instance(
                img_float, predict_fn, top_labels=len(PATHOLOGIES),
                hide_color=0, num_samples=1000, random_seed=42)
            if class_idx in explanation.local_exp:
                segments = explanation.segments
                exp_map = dict(explanation.local_exp[class_idx])
                attr_map = np.zeros((224, 224), dtype=np.float32)
                for seg_id, weight in exp_map.items():
                    attr_map[segments == seg_id] = max(weight, 0)
                if attr_map.max() > 0:
                    attr_map = attr_map / attr_map.max()
            else:
                attr_map = np.zeros((224, 224))
        except Exception:
            attr_map = np.zeros((224, 224))
        x1, y1, x2, y2 = get_box(row, orig_w, orig_h)
        results.append((attr_map, x1, y1, x2, y2))
    return results


def sweep_thresholds(maps, thresholds):
    """For a list of (attr_map, x1,y1,x2,y2), compute mean IoU at each threshold."""
    means = []
    for thresh in thresholds:
        ious = [compute_iou_at_threshold(m, x1, y1, x2, y2, thresh)
                for m, x1, y1, x2, y2 in maps]
        means.append(np.mean(ious))
    return means


def plot_sweep(results_dict, thresholds, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, means in results_dict.items():
        ax.plot(thresholds, means, "o-", linewidth=2,
                label=method, color=COLORS[method])
        for x, y in zip(thresholds, means):
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    ax.axvline(0.15, color="red", linestyle="--", linewidth=1,
               label="Default threshold (0.15)")
    ax.set_xlabel("IoU Threshold (top-X% of activation)", fontsize=11)
    ax.set_ylabel("Mean IoU", fontsize=11)
    ax.set_title("IoU Threshold Sensitivity — Grad-CAM vs LIME vs SHAP",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    model = load_model(CKPT_PATH, device)
    bbox_df = load_bbox_df(DATA_ROOT)
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]
    print(f"Bounding box records: {len(bbox_df)}")

    # Collect attribution maps for each method
    print("\nCollecting Grad-CAM maps...")
    gradcam_maps = collect_gradcam_maps(model, bbox_df, image_index, device)

    print("\nCollecting SHAP maps...")
    shap_maps = collect_shap_maps(model, bbox_df, image_index, device)

    print("\nCollecting LIME maps (this takes ~1.5 hours)...")
    lime_maps = collect_lime_maps(model, bbox_df, image_index, device)

    # Sweep thresholds
    print("\nSweeping thresholds...")
    results = {
        "Grad-CAM": sweep_thresholds(gradcam_maps, THRESHOLDS),
        "LIME":     sweep_thresholds(lime_maps,     THRESHOLDS),
        "SHAP":     sweep_thresholds(shap_maps,     THRESHOLDS),
    }

    # Save CSV
    rows = []
    for method, means in results.items():
        for thresh, mean_iou in zip(THRESHOLDS, means):
            rows.append({"Method": method, "Threshold": thresh, "Mean_IoU": round(mean_iou, 4)})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    print(f"Saved -> {RESULTS_DIR / 'threshold_sweep.csv'}")

    # Print table
    print("\n" + "=" * 55)
    print("IoU Threshold Sensitivity Results")
    print("=" * 55)
    header = f"{'Threshold':<12}" + "".join(f"{m:<14}" for m in results)
    print(header)
    print("-" * 55)
    for i, thresh in enumerate(THRESHOLDS):
        marker = " <-- default" if thresh == 0.15 else ""
        row = f"{int(thresh*100)}%{'':<10}" + "".join(
            f"{results[m][i]:<14.4f}" for m in results)
        print(row + marker)
    print("=" * 55)

    # Plot
    plot_sweep(results, THRESHOLDS, RESULTS_DIR / "threshold_sweep.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
