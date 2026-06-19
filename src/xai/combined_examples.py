"""
combined_examples.py
Creates five combined PNGs per pathology showing, side by side:
  Original X-ray | Grad-CAM | LIME | SHAP
with the ground-truth bounding box drawn on all panels.

Images are selected by computing (gradcam_iou + lime_iou + shap_iou) / 3
for every image that appears in all three existing metrics CSVs, then taking
the 5 images with the highest average IoU per pathology.

Outputs:
  results/combined_examples/<Pathology>_top1_combined.png
  ...
  results/combined_examples/<Pathology>_top5_combined.png
  (40 PNGs total — 5 per pathology x 8 pathologies)

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
TOP_N        = 5   # images selected per pathology
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
    img_rgb = np.array(pil_224)                        # uint8 (224,224,3)
    img_float = img_rgb.astype(np.float64) / 255.0
    explanation = explainer.explain_instance(
        img_float, predict_fn, top_labels=len(PATHOLOGIES),
        hide_color=0, num_samples=LIME_SAMPLES, random_seed=42)

    if class_idx in explanation.local_exp:
        _, mask = explanation.get_image_and_mask(
            class_idx, positive_only=True, num_features=10, hide_rest=False)
        # Green-blend positive superpixels onto original image, same as lime_exp.py
        base = img_rgb.astype(np.float32) / 255.0
        highlighted = base.copy()
        highlighted[mask == 1] = highlighted[mask == 1] * 0.5 + np.array([0, 0.8, 0]) * 0.5
        overlay_f = mark_boundaries(highlighted, mask)
        overlay = (np.clip(overlay_f, 0, 1) * 255).astype(np.uint8)
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
    # Blend SHAP heatmap with original X-ray so anatomy stays visible, same as shap_exp.py
    img_float = np.array(pil_224).astype(np.float32) / 255.0
    heatmap = plt.cm.RdBu_r(attr_map)[:, :, :3]
    blended = np.clip(img_float * 0.5 + heatmap * 0.5, 0, 1)
    overlay = (blended * 255).astype(np.uint8)
    overlay = draw_box(overlay, x1, y1, x2, y2)
    return overlay


def build_background(bbox_df, image_index, device, n=20):
    sample = bbox_df.sample(n=min(n, len(bbox_df)), random_state=42)
    tensors = []
    for _, row in sample.iterrows():
        pil = Image.open(image_index[row["Image Index"]]).convert("RGB")
        tensors.append(TRANSFORM(pil))
    return torch.stack(tensors).to(device)


def make_combined_figure(original, gradcam_img, lime_img, shap_img,
                         pathology, out_path, subtitle=""):
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
    suptitle = f"{pathology} — XAI Method Comparison (yellow = ground-truth box)"
    if subtitle:
        suptitle += f"\n{subtitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out_path}")


def load_avg_iou_ranking():
    """
    Join the three pre-computed metrics CSVs and compute per-image average IoU.
    Returns a DataFrame with columns: image, pathology, gradcam_iou, lime_iou,
    shap_iou, avg_iou — one row per image that appears in all three CSVs.
    """
    gc   = pd.read_csv(RESULTS_DIR / "gradcam_metrics.csv")[["image", "pathology", "iou"]].rename(columns={"iou": "gradcam_iou"})
    lime = pd.read_csv(RESULTS_DIR / "lime_metrics.csv")[["image", "pathology", "iou"]].rename(columns={"iou": "lime_iou"})
    shap = pd.read_csv(RESULTS_DIR / "shap_metrics.csv")[["image", "pathology", "iou"]].rename(columns={"iou": "shap_iou"})

    merged = gc.merge(lime, on=["image", "pathology"]).merge(shap, on=["image", "pathology"])
    merged["avg_iou"] = (merged["gradcam_iou"] + merged["lime_iou"] + merged["shap_iou"]) / 3.0
    return merged


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pre-computed IoU scores and rank images ─────────────────────────
    print("Loading metrics CSVs...")
    ranking = load_avg_iou_ranking()
    print(f"  images in all three metrics: {len(ranking)}")

    # ── Index images on disk ─────────────────────────────────────────────────
    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")
    ranking = ranking[ranking["image"].isin(image_index.keys())]

    model = load_model(CKPT_PATH, device)

    bbox_df = load_bbox_df(DATA_ROOT)
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]

    print("Building SHAP background...")
    background = build_background(bbox_df, image_index, device, SHAP_BG_SIZE)

    # ── Per pathology: select top-N by average IoU and generate figures ───────
    for pathology in BOXED_PATHOLOGIES:
        subset = ranking[ranking["pathology"] == pathology]
        if len(subset) == 0:
            print(f"\nSkipping {pathology} — no records in merged metrics")
            continue

        top_n = subset.nlargest(TOP_N, "avg_iou").reset_index(drop=True)

        print(f"\n{pathology} — top-{TOP_N} images by avg IoU:")
        for _, r in top_n.iterrows():
            print(f"  {r['image']}  avg={r['avg_iou']:.4f}  "
                  f"(GC={r['gradcam_iou']:.3f}  LIME={r['lime_iou']:.3f}  SHAP={r['shap_iou']:.3f})")

        class_idx = PATHOLOGIES.index(pathology)

        for rank, (_, rec) in enumerate(top_n.iterrows(), start=1):
            fname = rec["image"]

            box_rows = bbox_df[(bbox_df["Image Index"] == fname) &
                               (bbox_df["Finding Label"] == pathology)]
            if len(box_rows) == 0:
                print(f"  WARNING: no bbox row for {fname}, skipping rank {rank}")
                continue
            box_row = box_rows.iloc[0]

            print(f"  [{rank}/{TOP_N}] {fname}  avg_iou={rec['avg_iou']:.4f}")
            pil_img = Image.open(image_index[fname]).convert("RGB")
            orig_w, orig_h = pil_img.size
            pil_224 = pil_img.resize((224, 224))
            tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
            x1, y1, x2, y2 = get_box(box_row, orig_w, orig_h)

            img_rgb_float = np.array(pil_224).astype(np.float32) / 255.0
            original_with_box = draw_box(np.array(pil_224), x1, y1, x2, y2)

            print("    running Grad-CAM...")
            gradcam_img = run_gradcam(model, tensor, class_idx, img_rgb_float, x1, y1, x2, y2)

            print("    running LIME (slowest step)...")
            lime_img = run_lime(model, device, pil_224, class_idx, x1, y1, x2, y2)

            print("    running SHAP...")
            shap_img = run_shap(model, device, background, tensor, class_idx, pil_224, x1, y1, x2, y2)

            subtitle = (f"avg IoU={rec['avg_iou']:.3f}  "
                        f"(GC={rec['gradcam_iou']:.3f}  LIME={rec['lime_iou']:.3f}  SHAP={rec['shap_iou']:.3f})")
            out_path = OUT_DIR / f"{pathology}_top{rank}_combined.png"
            make_combined_figure(original_with_box, gradcam_img, lime_img, shap_img,
                                 pathology, out_path, subtitle=subtitle)

    print(f"\nAll {TOP_N * len(BOXED_PATHOLOGIES)} combined figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
