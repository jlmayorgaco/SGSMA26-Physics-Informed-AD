from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.filesystem import ensure_dir


def save_figure(fig, path: Path, dpi: int = 220, close: bool = True) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def add_event_span(ax, start_time: float, end_time: float, alpha: float = 0.15) -> None:
    ax.axvline(start_time, linestyle="--", linewidth=1.0)
    ax.axvline(end_time, linestyle="--", linewidth=1.0)
    ax.axvspan(start_time, end_time, alpha=alpha)


def style_axis(ax, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def maybe_add_legend(ax, loc: str = "best", ncol: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, ncol=ncol)


def style_axes(axes: Iterable, xlabel: Optional[str] = None) -> None:
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if xlabel:
            ax.set_xlabel(xlabel)