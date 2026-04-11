"""Plot helpers for ablation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_ablation_heatmap(table: pd.DataFrame, out_path: Path) -> None:
    """Save a 4x4 heatmap of real Macro-F1 values."""

    import matplotlib.pyplot as plt

    pivot = table.pivot(index="estimator", columns="classifier", values="macro_f1_real")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("Macro-F1 on real events")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save a 9x9 confusion matrix image."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="magma")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

