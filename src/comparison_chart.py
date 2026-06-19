"""
comparison_chart.py
Generates publication-quality comparison charts for showing
Grad-CAM, LIME, and SHAP results side by side.

Produces:
  results/comparison_pointing_game.png   — pointing game bar chart
  results/comparison_iou.png             — IoU bar chart
  results/comparison_combined.png        — both metrics in one figure
  results/comparison_radar.png           — radar chart per method

Run (from D:\\thesis\\src):
    python comparison_chart.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================ CONFIG ========================================
RESULTS_DIR = Path(r"D:\thesis\results")
# ===========================================================================

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]

# Colors for the three methods — colorblind-friendly palette
COLORS = {
    "Grad-CAM": "#2196F3",   # blue
    "LIME":     "#FF9800",   # orange
    "SHAP":     "#4CAF50",   # green
}


def load_summaries():
    """Load the three summary CSVs and extract per-pathology rows only."""
    summaries = {}
    files = {
        "Grad-CAM": RESULTS_DIR / "gradcam_summary.csv",
        "LIME":     RESULTS_DIR / "lime_summary.csv",
        "SHAP":     RESULTS_DIR / "shap_summary.csv",
    }
    for method, path in files.items():
        df = pd.read_csv(path, index_col=0)
        # Drop the MEAN row for per-pathology charts
        df = df[df.index != "MEAN"]
        summaries[method] = df
    return summaries


def plot_grouped_bar(summaries, metric, ylabel, title, out_path, show_target=False):
    """Grouped bar chart: pathologies on x-axis, one bar per method."""
    n_paths = len(PATHOLOGIES)
    n_methods = len(summaries)
    width = 0.25
    x = np.arange(n_paths)

    fig, ax = plt.subplots(figsize=(13, 6))

    for j, (method, df) in enumerate(summaries.items()):
        values = []
        for p in PATHOLOGIES:
            if p in df.index:
                values.append(float(df.loc[p, metric]))
            else:
                values.append(0.0)
        offset = (j - 1) * width
        bars = ax.bar(x + offset, values, width,
                      label=method, color=COLORS[method],
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7,
                     rotation=90, label_type="edge")

    if show_target:
        ax.axhline(0.5, color="red", linestyle="--",
                   linewidth=1, label="0.5 reference")

    ax.set_xticks(x)
    ax.set_xticklabels(PATHOLOGIES, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_combined(summaries, out_path):
    """Two subplots stacked vertically: pointing game on top, IoU on bottom."""
    n_paths = len(PATHOLOGIES)
    width = 0.25
    x = np.arange(n_paths)

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    metrics = [
        ("pointing_game", "Pointing Game Accuracy", "A  —  Pointing Game"),
        ("mean_iou",      "Mean IoU",               "B  —  Mean IoU"),
    ]

    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        for j, (method, df) in enumerate(summaries.items()):
            values = []
            for p in PATHOLOGIES:
                if p in df.index:
                    values.append(float(df.loc[p, metric]))
                else:
                    values.append(0.0)
            offset = (j - 1) * width
            bars = ax.bar(x + offset, values, width,
                          label=method, color=COLORS[method],
                          edgecolor="white", linewidth=0.5)
            ax.bar_label(bars, fmt="%.2f", padding=2,
                         fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(PATHOLOGIES, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subtitle, fontsize=12, fontweight="bold", loc="left")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "XAI Localization Performance: Grad-CAM vs LIME vs SHAP\n"
        "NIH ChestX-ray14 — Bounding Box Evaluation Set",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_mean_summary(summaries, out_path):
    """Simple mean score bar chart — clean summary for thesis intro figure."""
    methods = list(summaries.keys())
    pg_means, iou_means = [], []

    for method, df in summaries.items():
        df_paths = df[df.index != "MEAN"]
        pg_means.append(float(df_paths["pointing_game"].mean()))
        iou_means.append(float(df_paths["mean_iou"].mean()))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, pg_means, width, label="Pointing Game",
                color=["#2196F3", "#FF9800", "#4CAF50"],
                edgecolor="white")
    b2 = ax.bar(x + width/2, iou_means, width, label="Mean IoU",
                color=["#90CAF9", "#FFCC80", "#A5D6A7"],
                edgecolor="white")

    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=10)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Mean Localization Scores by XAI Method",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Manual legend for metric type
    patch1 = mpatches.Patch(color="grey",       label="Pointing Game (dark)")
    patch2 = mpatches.Patch(color="lightgrey",  label="IoU (light)")
    ax.legend(handles=[patch1, patch2], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_radar(summaries, out_path):
    """Radar / spider chart — one spoke per pathology, one line per method."""
    categories = PATHOLOGIES
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw=dict(polar=True))

    metric_info = [
        ("pointing_game", "Pointing Game"),
        ("mean_iou",      "Mean IoU"),
    ]

    for ax, (metric, title) in zip(axes, metric_info):
        for method, df in summaries.items():
            values = []
            for p in categories:
                if p in df.index:
                    values.append(float(df.loc[p, metric]))
                else:
                    values.append(0.0)
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2,
                    label=method, color=COLORS[method])
            ax.fill(angles, values, alpha=0.08, color=COLORS[method])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=12, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("XAI Method Comparison — Radar Chart",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    print("Loading summary CSVs...")
    summaries = load_summaries()

    print("Generating charts...")

    plot_grouped_bar(
        summaries,
        metric="pointing_game",
        ylabel="Pointing Game Accuracy",
        title="Pointing Game Accuracy by Pathology and XAI Method",
        out_path=RESULTS_DIR / "comparison_pointing_game.png",
    )

    plot_grouped_bar(
        summaries,
        metric="mean_iou",
        ylabel="Mean IoU",
        title="Mean IoU by Pathology and XAI Method",
        out_path=RESULTS_DIR / "comparison_iou.png",
    )

    plot_combined(
        summaries,
        out_path=RESULTS_DIR / "comparison_combined.png",
    )

    plot_mean_summary(
        summaries,
        out_path=RESULTS_DIR / "comparison_mean_summary.png",
    )

    plot_radar(
        summaries,
        out_path=RESULTS_DIR / "comparison_radar.png",
    )

    print("\nAll charts saved to", RESULTS_DIR)
    print("\nFiles for your thesis:")
    print("  comparison_combined.png     — main results figure (use this one)")
    print("  comparison_mean_summary.png — clean overview for introduction/abstract")
    print("  comparison_radar.png        — radar chart for discussion section")
    print("  comparison_pointing_game.png — standalone pointing game detail")
    print("  comparison_iou.png           — standalone IoU detail")


if __name__ == "__main__":
    main()
