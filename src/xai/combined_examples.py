"""
combined_examples.py
Creates one combined PNG per pathology showing, side by side:
  Original X-ray | Grad-CAM | LIME | SHAP
with the ground-truth bounding box drawn on all three XAI panels.

This reuses the same image (and same ground-truth box) across all four
panels so the comparison is apples-to-apples — same patient, same box,
three different explanations.

Outputs:
  results/combined_examples/<Pathology>_combined.png   (one per pathology, 8 total)

Run (from D:\\thesis\\src\\xai):
    python combined_examples.py
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
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import PATHOLOGIES, build_image_index

# ============================ CONFIG ========================================
DATA_ROOT    = r"D:\archive"
CKPT_PATH    = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR  = Path(r"D:\thesis\results")
OUT_DIR      = RESULTS_DIR / "combined_examples"
LIME_SAMPLES = 1000
SHAP_BG_SIZE = 20
IoU_THRESHOLD = 0.15
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
    return df


def get_box(row, orig_w, orig_h, target=224):
    sx = target / orig_w
    sy = target / orig_h
    x1 = max(0,      int(row["Bbox [x"] * sx))
    y1 = max(0,      int(row["y"] * sy))
    x2 = min(target, int((row["Bbox [x"] + row["w"]) * sx))
    y2 = min(target, int((row["y"] + row["h]"]) * sy))
    return x1, y1, x2, y2


def draw_box(img_array, x1, y1, x2, y2, color=(255, 255, 0)):
    """Draw a yellow rectangle outline on a copy of the image array."""
    out = img_array.copy()
    out[y1:y2, x1:x1+2] = color
    out[y1:y2, x2:x2+2] = color
    out[y1:y1+2, x1:x2] = color
    out[y2:y2+2, x1:x2] = color
    return out


def run_gradcam(model, tensor, class_idx, img_rgb_float, x1, y1, x2, y2):
    target_layer = [model.features.denseblock4.denselayer16.conv2]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(class_idx)]
    cam = cam_extractor(input_tensor=tensor, targets=targets)[0]
    cam = np.float32(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    overlay = show_cam_on_image(img_rgb_float, cam, use_rgb=True)
    overlay = draw_box(overlay, x1, y1, x2, y2)
    return overlay


def run_lime(model, device, pil_224, class_idx, x1, y1, x2, y2):
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
    img_float = np.array(pil_224).astype(np.float64) / 255.0
    explanation = explainer.explain_instance(
        img_float, predict_fn, top_labels=len(PATHOLOGIES),
        hide_color=0, num_samples=LIME_SAMPLES, random_seed=42)

    if class_idx in explanation.local_exp:
        temp, mask = explanation.get_image_and_mask(
            class_idx, positive_only=True, num_features=10, hide_rest=False)
        overlay = mark_boundaries(temp / 255.0, mask)
        overlay = (overlay * 255).astype(np.uint8)
    else:
        overlay = np.array(pil_224)
    overlay = draw_box(overlay, x1, y1, x2, y2)
    return overlay


def run_shap(model, device, background, tensor, class_idx, pil_224, x1, y1, x2, y2):
    explainer = GradientShap(model)
    tensor_grad = tensor.clone().requires_grad_(True)
    attr = explainer.attribute(tensor_grad, baselines=background,
                               target=class_idx, n_samples=10, stdevs=0.09)
    attr_map = np.abs(attr[0].detach().cpu().numpy()).mean(axis=0)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    overlay = plt.cm.RdBu_r(attr_map)[:, :, :3]
    overlay = (overlay * 255).astype(np.uint8)
    overlay = draw_box(overlay, x1, y1, x2, y2)
    return overlay


def build_background(bbox_df, image_index, device, n=20):
    sample = bbox_df.sample(n=min(n, len(bbox_df)), random_state=42)
    tensors = []
    for _, row in sample.iterrows():
        pil = Image.open(image_index[row["Image Index"]]).convert("RGB")
        tensors.append(TRANSFORM(pil))
    return torch.stack(tensors).to(device)


def make_combined_figure(original, gradcam_img, lime_img, shap_img, pathology, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        (original,    "Original X-ray"),
        (gradcam_img, "Grad-CAM"),
        (lime_img,    "LIME"),
        (shap_img,    "SHAP (GradientSHAP)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{pathology} — XAI Method Comparison (yellow = ground-truth box)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    model = load_model(CKPT_PATH, device)
    bbox_df = load_bbox_df(DATA_ROOT)
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]

    print("Building SHAP background...")
    background = build_background(bbox_df, image_index, device, SHAP_BG_SIZE)

    # Pick ONE representative image per pathology — the first one in the bbox list
    for pathology in BOXED_PATHOLOGIES:
        subset = bbox_df[bbox_df["Finding Label"] == pathology]
        if len(subset) == 0:
            print(f"  skipping {pathology} — no bbox records found")
            continue
        row = subset.iloc[0]
        fname = row["Image Index"]
        class_idx = PATHOLOGIES.index(pathology)

        print(f"\nProcessing {pathology} ({fname})...")
        pil_img = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size
        pil_224 = pil_img.resize((224, 224))
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
        x1, y1, x2, y2 = get_box(row, orig_w, orig_h)

        img_rgb_float = np.array(pil_224).astype(np.float32) / 255.0
        original_with_box = draw_box(np.array(pil_224), x1, y1, x2, y2)

        print("  running Grad-CAM...")
        gradcam_img = run_gradcam(model, tensor, class_idx, img_rgb_float, x1, y1, x2, y2)

        print("  running LIME (slowest step)...")
        lime_img = run_lime(model, device, pil_224, class_idx, x1, y1, x2, y2)

        print("  running SHAP...")
        shap_img = run_shap(model, device, background, tensor, class_idx, pil_224, x1, y1, x2, y2)

        out_path = OUT_DIR / f"{pathology}_combined.png"
        make_combined_figure(original_with_box, gradcam_img, lime_img, shap_img,
                             pathology, out_path)

    print("\nAll combined comparison images saved to", OUT_DIR)


if __name__ == "__main__":
    main()
