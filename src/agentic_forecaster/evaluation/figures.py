"""Standard figure set generated from real run artefacts.

All figures are produced from actual run data (predictions, metrics,
attention weights, attributions) and record the source run id so provenance
is never lost.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("agentic_forecaster.evaluation.figures")


def reliability_diagram(y_true, p, path: Path, title: str) -> Path:
    bins = np.linspace(0, 1, 11)
    accs, confs = [], []
    for i in range(10):
        mask = (p >= bins[i]) & ((p < bins[i + 1]) if i < 9 else (p <= bins[i + 1]))
        if mask.sum() == 0:
            continue
        accs.append(float(y_true[mask].mean()))
        confs.append(float(p[mask].mean()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(confs, accs, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability diagram — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrix(y_true, p, path: Path, title: str) -> Path:
    pred = (p >= 0.5).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, q in zip(y_true, pred):
        cm[int(t), int(q)] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["down", "up"])
    ax.set_yticks([0, 1], ["down", "up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def paper_comparison(comparison, path: Path) -> Path:
    models, paper, recon = [], [], []
    for row in comparison or []:
        ref = row.get("accuracy_paper_reference")
        rep = row.get("accuracy_reconstructed")
        if ref is None and rep is None:
            continue
        models.append(row.get("model"))
        paper.append(ref if ref is not None else np.nan)
        recon.append(rep if rep is not None else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    if models:
        x = np.arange(len(models))
        ax.bar(x - 0.2, paper, 0.4, label="PAPER REFERENCE")
        ax.bar(x + 0.2, recon, 0.4, label="reconstruction")
        ax.set_xticks(x, models, rotation=30, ha="right")
        ax.set_ylabel("Accuracy")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No comparison data", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Paper reference vs reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def precision_at_3_figure(p3_rows, path: Path) -> Path:
    folds = [r.get("fold") for r in (p3_rows or [])]
    up = [r.get("precision_at_3_up", 0.0) for r in (p3_rows or [])]
    down = [r.get("precision_at_3_down", 0.0) for r in (p3_rows or [])]
    fig, ax = plt.subplots(figsize=(6, 4))
    if folds:
        x = np.arange(len(folds))
        ax.bar(x - 0.2, up, 0.4, label="P@3 Up")
        ax.bar(x + 0.2, down, 0.4, label="P@3 Down")
        ax.set_xticks(x, folds)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Precision")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No P@3 data", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Cross-sectional Precision@3 by date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_run_figures(
    predictions: pd.DataFrame,
    comparison: list[dict],
    p3_rows: list[dict],
    out_dir: Path,
    run_id: str,
) -> list[str]:
    """Write the standard figure set plus a provenance sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if predictions is not None and not predictions.empty:
        y = predictions["y"].to_numpy(dtype=float)
        p_cal = predictions["calibrated_p_up"].to_numpy(dtype=float)
        written.append(str(reliability_diagram(
            y, p_cal, out_dir / "reliability_diagram.png", f"run {run_id}")))
        written.append(str(confusion_matrix(
            y, p_cal, out_dir / "confusion_matrix.png", f"run {run_id}")))
    written.append(str(paper_comparison(comparison, out_dir / "paper_comparison.png")))
    written.append(str(precision_at_3_figure(p3_rows, out_dir / "precision_at_3.png")))

    (out_dir / "figures_manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "provenance": "generated_from_real_run_artifacts",
        "figures": [Path(p).name for p in written],
    }, indent=2))
    logger.info("Wrote %d figure(s) to %s", len(written), out_dir)
    return written
