"""
lime_exp.py
Runs LIME on all bounding-box images in NIH ChestX-ray14 and computes
the same two localization metrics as gradcam.py so results are directly
comparable:

  1. Pointing Game  — does the highest-weight superpixel centroid fall inside box?
  2. IoU            — overlap between positive LIME mask and the GT box

Outputs:
  results/lime_metrics.csv       per-image scores
  results/lime_summary.csv       per-pathology mean scores  (thesis Table 3)
  results/lime_examples/         explanation overlay images for thesis figures

Run (from D:\\thesis\\src\\xai):
    python lime_exp.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms

from dataset import PATHOLOGIES, build_image_index

# ============================ CONFIG ========================================
DATA_ROOT     = r"D:\archive"
CKPT_PATH     = r"D:\thesis\src\checkpoints\densenet121_best.pt"
RESULTS_DIR   = Path(r"D:\thesis\results")
EXAMPLES_DIR  = RESULTS_DIR / "lime_examples"
NUM_SAMPLES   = 1000   # LIME perturbation samples — higher = more accurate but slower
                        # 1000 is standard; drop to 500 if too slow
NUM_FEATURES  = 10     # number of superpixel regions LIME returns
NUM_EXAMPLES  = 5      # overlay images saved per pathology
# ===========================================================================

BOXED_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])


# ── helpers ──────────────────────────────────────────────────────────────────

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


def make_predict_fn(model, device):
    """Return a function that LIME can call: takes (N, H, W, 3) uint8 -> (N, 14) probs."""
    def predict(images):
        # images: numpy array (N, 224, 224, 3) float64 in [0,1] from LIME
        batch = []
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)
            pil = Image.fromarray(img_uint8)
            tensor = TENSOR_TRANSFORM(pil)
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    return predict


def scale_box(row, target=224):
    """BBox columns: 'Bbox [x', 'y', 'w', 'h]' — already at original resolution.
    We resize images to 224x224 so we need to scale."""
    # We load original image to get its size in main loop
    # This function receives pre-scaled values
    return row


def get_lime_mask(explanation, class_idx, num_features):
    """Extract positive-weight superpixel mask for the target class."""
    segments = explanation.segments  # (224, 224) int array of superpixel IDs
    # Get top superpixels with positive contribution
    exp_map = dict(explanation.local_exp[class_idx])
    # Build mask: 1 where superpixel has positive weight
    mask = np.zeros(segments.shape, dtype=np.uint8)
    for seg_id, weight in exp_map.items():
        if weight > 0:
            mask[segments == seg_id] = 1
    return mask, segments, exp_map


def pointing_game_lime(exp_map, segments):
    """Pointing game for LIME: centroid of highest-weight superpixel inside box?"""
    if not exp_map:
        return 0, (0, 0)
    best_seg = max(exp_map, key=lambda k: exp_map[k])
    ys, xs = np.where(segments == best_seg)
    if len(ys) == 0:
        return 0, (0, 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    return cx, cy  # return centroid, check against box in main loop


def compute_iou(mask, x1, y1, x2, y2):
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    intersection = (mask & box_mask).sum()
    union = (mask | box_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def save_example(img_rgb, explanation, class_idx, x1, y1, x2, y2,
                 num_features, out_path):
    """Save original | LIME explanation with GT box overlay."""
    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    # Blend LIME mask on top of original X-ray so the anatomy stays visible
    img_float = img_rgb.astype(np.float32) / 255.0
    highlighted = img_float.copy()
    highlighted[mask == 1] = highlighted[mask == 1] * 0.5 + np.array([0, 0.8, 0]) * 0.5
    overlay = mark_boundaries(highlighted, mask)

    # Draw GT box
    overlay_copy = overlay.copy()
    overlay_copy[y1:y2, x1:x1+2] = [1, 1, 0]
    overlay_copy[y1:y2, x2:x2+2] = [1, 1, 0]
    overlay_copy[y1:y1+2, x1:x2] = [1, 1, 0]
    overlay_copy[y2:y2+2, x1:x2] = [1, 1, 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    axes[1].imshow(overlay_copy)
    axes[1].set_title("LIME + GT box (yellow)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Indexing images...")
    image_index = build_image_index(DATA_ROOT)
    print(f"  found {len(image_index)} PNG files")

    model = load_model(CKPT_PATH, device)
    predict_fn = make_predict_fn(model, device)

    bbox_df = load_bbox_df(DATA_ROOT)
    bbox_df = bbox_df[bbox_df["Image Index"].isin(image_index.keys())]

    explainer = lime_image.LimeImageExplainer(random_state=42)

    records = []
    example_counts = {p: 0 for p in BOXED_PATHOLOGIES}
    total = len(bbox_df)

    for i, (_, row) in enumerate(bbox_df.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total}")

        fname     = row["Image Index"]
        pathology = row["Finding Label"]
        class_idx = PATHOLOGIES.index(pathology)

        # Load and resize image
        pil_img  = Image.open(image_index[fname]).convert("RGB")
        orig_w, orig_h = pil_img.size
        pil_224  = pil_img.resize((224, 224))
        img_array = np.array(pil_224)           # (224,224,3) uint8

        # Scale GT box to 224×224
        sx = 224 / orig_w
        sy = 224 / orig_h
        x1 = max(0,   int(row["Bbox [x"] * sx))
        y1 = max(0,   int(row["y"] * sy))
        h  = row["h]"]
        x2 = min(224, int((row["Bbox [x"] + row["w"]) * sx))
        y2 = min(224, int((row["y"] + h) * sy))

        # Run LIME
        # LIME expects float [0,1] image
        img_float = img_array.astype(np.float64) / 255.0
        explanation = explainer.explain_instance(
            img_float,
            predict_fn,
            top_labels=len(PATHOLOGIES),
            hide_color=0,
            num_samples=NUM_SAMPLES,
            random_seed=42,
        )

        # Metrics
        if class_idx not in explanation.local_exp:
            records.append({
                "image": fname, "pathology": pathology,
                "pointing_game": 0, "iou": 0.0,
            })
            continue

        mask, segments, exp_map = get_lime_mask(explanation, class_idx, NUM_FEATURES)
        cx, cy = pointing_game_lime(exp_map, segments)
        pg  = int(x1 <= cx <= x2 and y1 <= cy <= y2)
        iou = compute_iou(mask, x1, y1, x2, y2)

        records.append({
            "image": fname, "pathology": pathology,
            "pointing_game": pg, "iou": iou,
        })

        # Save example overlays
        if example_counts[pathology] < NUM_EXAMPLES:
            img_rgb = np.array(pil_224)
            out_path = EXAMPLES_DIR / f"{pathology}_{example_counts[pathology]+1}.png"
            try:
                save_example(img_rgb, explanation, class_idx,
                             x1, y1, x2, y2, NUM_FEATURES, out_path)
                example_counts[pathology] += 1
            except Exception:
                pass  # don't let a save failure break the whole run

    # ── aggregate ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / "lime_metrics.csv", index=False)

    summary = results_df.groupby("pathology").agg(
        n_images=("image", "count"),
        pointing_game=("pointing_game", "mean"),
        mean_iou=("iou", "mean"),
    ).round(4)
    summary.loc["MEAN"] = summary.mean()

    print("\n" + "=" * 55)
    print("LIME Localization Results")
    print("=" * 55)
    print(summary.to_string())
    print("=" * 55)
    summary.to_csv(RESULTS_DIR / "lime_summary.csv")
    print(f"\nSaved -> {RESULTS_DIR / 'lime_summary.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'lime_metrics.csv'}")
    print(f"Example images -> {EXAMPLES_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
