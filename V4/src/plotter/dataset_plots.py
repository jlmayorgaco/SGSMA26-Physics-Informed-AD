from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotter.common import save_figure, style_axis


def plot_dataset_correlation_heatmap(corr_df: pd.DataFrame, title: str, out_path: Path, dpi: int = 220) -> None:
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.to_numpy(dtype=float))
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    style_axis(ax, title=title)
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)


def plot_missing_raster(
    time: np.ndarray,
    missing_matrix: np.ndarray,
    bus_labels: list[str],
    out_path: Path,
    dpi: int = 220,
) -> None:
    if missing_matrix.size == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(
        missing_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=[time[0], time[-1], 0, len(bus_labels)],
        origin="lower",
    )
    ax.set_yticks(np.arange(len(bus_labels)) + 0.5)
    ax.set_yticklabels(bus_labels)
    style_axis(ax, title="Missing-data raster", xlabel="Time [s]", ylabel="Bus")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)