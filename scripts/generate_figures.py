#!/usr/bin/env python3
"""Generate the final publication/reproduction figures.

Usage:
    python scripts/generate_figures.py --run-dir RUN_DIR [--out figures/]

Figures produced:
  - reliability_diagram.png
  - training_history.png
  - confusion_matrix.png
  - paper_comparison.png
  - attention_visualization.png
  - shap_summary.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def reliability_diagram(y_true, proba, path):
    bins = np.linspace(0, 1, 11)
    accs, confs, counts = [], [], []
    for i in range(10):
        if i < 9:
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
        else:
            mask = (proba >= bins[i]) & (proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        accs.append(y_true[mask].mean())
        confs.append(proba[mask].mean())
        counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(confs, accs, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_history(history, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix(y_true, y_pred, path):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["down", "up"])
    ax.set_yticks([0, 1], ["down", "up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paper_comparison_fig(comparison_csv, path):
    import csv

    with open(comparison_csv) as f:
        rows = list(csv.DictReader(f))
    models = []
    paper_vals = []
    recon_vals = []
    for r in rows:
        if r["metric"] == "accuracy" and r["paper_reference"] and r["reconstructed_run"]:
            models.append(r["model"])
            paper_vals.append(float(r["paper_reference"]))
            recon_vals.append(float(r["reconstructed_run"]))
    if not models:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No comparison data yet", ha="center")
        ax.axis("off")
    else:
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, paper_vals, 0.4, label="paper")
        ax.bar(x + 0.2, recon_vals, 0.4, label="reconstructed")
        ax.set_xticks(x, models, rotation=30, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Paper reference vs reconstruction")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attention_viz(weights, path):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(weights.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Day in lookback")
    ax.set_title("Attention weights")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_summary(top_features, path):
    names = [f["feature"] for f in top_features]
    vals = [f["mean_abs_shap"] for f in top_features]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "figures"))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    json.loads((run_dir / "metrics.json").read_text())
    explanation = json.loads((run_dir / "explanation.json").read_text())

    # Reliability + confusion need predictions; generate from metrics context
    # For the committed figures we use the demo run's stored arrays if present,
    # otherwise produce clearly-labelled schematic figures.
    pred_path = run_dir / "test_predictions.npz"
    if pred_path.exists():
        z = np.load(pred_path)
        y_true, proba = z["y"], z["p"]
        reliability_diagram(y_true, proba, out / "reliability_diagram.png")
        confusion_matrix(y_true, (proba >= 0.5).astype(int), out / "confusion_matrix.png")

    hist_path = run_dir / "training_history.json"
    if hist_path.exists():
        training_history(json.loads(hist_path.read_text()), out / "training_history.png")

    comparison_csv = REPO_ROOT / "results" / "paper_reproduction" / "paper_comparison.csv"
    if comparison_csv.exists():
        paper_comparison_fig(comparison_csv, out / "paper_comparison.png")

    att_path = run_dir / "attention_weights.npy"
    if att_path.exists():
        attention_viz(np.load(att_path), out / "attention_visualization.png")

    top = explanation.get("top_features", [])
    if top:
        shap_summary(top, out / "shap_summary.png")

    print(f"Figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
