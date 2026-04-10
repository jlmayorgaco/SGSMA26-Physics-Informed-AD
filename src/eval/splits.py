"""Contiguous 70 / 15 / 15 time-block split with a 1-window buffer.

The buffer prevents any feature window (window_sec seconds wide) from straddling
a split boundary — extracting features at a boundary onset would pull data from
the wrong split, creating leakage.

Usage
-----
    from src.eval.splits import make_splits

    splits = make_splits(df, window_sec=3.0, fps=30.0)
    train_idx = splits["train"]   # array of integer row indices
    val_idx   = splits["val"]
    test_idx  = splits["test"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    window_sec: float = 3.0,
    fps: float = 30.0,
) -> dict[str, np.ndarray]:
    """Compute contiguous train / val / test row-index arrays.

    Args:
        df:          Merged DataFrame with a TIMESTAMP column.
        train_frac:  Fraction of rows for training (default 0.70).
        val_frac:    Fraction of rows for validation (default 0.15).
                     test_frac = 1 - train_frac - val_frac.
        window_sec:  Feature window half-width in seconds (buffer = window_sec
                     frames on each side of a split boundary).
        fps:         Sampling rate (used to convert window_sec to frame count).

    Returns:
        dict with keys "train", "val", "test", each an ndarray of integer
        row indices into df (contiguous, no gaps except buffers).
    """
    n = len(df)
    buf = int(round(fps * window_sec))  # 1 full window = buffer width

    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    # Enforce minimum split sizes
    if train_end <= buf:
        raise ValueError(f"train split ({train_end}) smaller than buffer ({buf})")
    if val_end - train_end <= 2 * buf:
        raise ValueError(f"val split ({val_end - train_end}) smaller than 2×buffer ({2 * buf})")
    if n - val_end <= buf:
        raise ValueError(f"test split ({n - val_end}) smaller than buffer ({buf})")

    train_idx = np.arange(0, train_end - buf, dtype=int)
    val_idx   = np.arange(train_end + buf, val_end - buf, dtype=int)
    test_idx  = np.arange(val_end + buf, n, dtype=int)

    return {"train": train_idx, "val": val_idx, "test": test_idx}
