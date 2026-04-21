from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def plot_event_timeline(
    timeline: pd.Index,
    true_event5: np.ndarray,
    pred_event5: np.ndarray,
    true_event7: np.ndarray,
    pred_event7: np.ndarray,
    output_path: Path,
    max_points: int = 8000,
) -> None:
    idx = _downsample_indices(len(timeline), max_points=max_points)
    x = timeline.to_numpy(dtype=float)[idx]
    t5 = np.asarray(true_event5, dtype=int)[idx]
    p5 = np.asarray(pred_event5, dtype=int)[idx]
    t7 = np.asarray(true_event7, dtype=int)[idx]
    p7 = np.asarray(pred_event7, dtype=int)[idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].step(x, t5, where="post", label="true e5", linewidth=1.4)
    axes[0].step(x, p5, where="post", label="pred e5", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("event5")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].step(x, t7, where="post", label="true e7", linewidth=1.4)
    axes[1].step(x, p7, where="post", label="pred e7", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("event7")
    axes[1].set_xlabel("timestamp (s)")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_event7_score_histogram(
    event7_score: np.ndarray,
    true_event7: np.ndarray,
    output_path: Path,
) -> None:
    scores = np.asarray(event7_score, dtype=float)
    labels = np.asarray(true_event7, dtype=int)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores[labels == 0], bins=80, alpha=0.55, label="true != 7", density=True)
    if np.any(labels == 1):
        ax.hist(scores[labels == 1], bins=60, alpha=0.70, label="true == 7", density=True)
    ax.set_xlabel("max event7 score")
    ax.set_ylabel("density")
    ax.set_title("Event7 score distribution")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_active_bus_count(
    active_bus_count: np.ndarray,
    true_event7: np.ndarray,
    output_path: Path,
) -> None:
    counts = np.asarray(active_bus_count, dtype=int)
    labels = np.asarray(true_event7, dtype=int)

    max_count = max(int(np.nanmax(counts)) if counts.size else 0, 1)
    bins = np.arange(0, max_count + 2, dtype=int)
    width = 0.35

    hist_non7, _ = np.histogram(counts[labels == 0], bins=bins)
    hist_7, _ = np.histogram(counts[labels == 1], bins=bins) if np.any(labels == 1) else (np.zeros_like(hist_non7), bins)
    x = np.arange(len(hist_non7))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2.0, hist_non7, width=width, label="true != 7", alpha=0.75)
    ax.bar(x + width / 2.0, hist_7, width=width, label="true == 7", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in range(len(hist_non7))])
    ax.set_xlabel("active buses in frame")
    ax.set_ylabel("count")
    ax.set_title("Active bus count distribution")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

