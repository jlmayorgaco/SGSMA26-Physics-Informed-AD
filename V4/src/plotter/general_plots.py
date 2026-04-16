from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.constants import TIMESTAMP_COLUMN
from src.plotter.common import save_figure, style_axis, maybe_add_legend


def plot_general_normal_operation_signal(
    df: pd.DataFrame,
    column: str,
    out_path: Path,
    dpi: int = 220,
) -> None:
    if column not in df.columns or df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df[TIMESTAMP_COLUMN], df[column], label=column)
    style_axis(ax, title=f"Normal operation | {column}", xlabel="Time [s]", ylabel=column)
    maybe_add_legend(ax)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)


def plot_general_histogram(
    df: pd.DataFrame,
    column: str,
    out_path: Path,
    dpi: int = 220,
) -> None:
    if column not in df.columns or df.empty:
        return

    values = df[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=100)
    style_axis(ax, title=f"Normal operation histogram | {column}", xlabel=column, ylabel="Count")

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)