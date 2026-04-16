from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.constants import ANGLE_COLUMNS


def as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def unwrap_if_angle(values: np.ndarray, column: str) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()

    if column not in ANGLE_COLUMNS:
        return y

    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y

    out = y.copy()
    out[mask] = np.rad2deg(np.unwrap(np.deg2rad(y[mask])))
    return out


def rolling_trend(y: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(3, int(window_samples) | 1)

    trend = y.rolling(
        window=window_samples,
        center=True,
        min_periods=max(3, window_samples // 5),
    ).median()

    trend = trend.interpolate(limit_direction="both")
    trend = trend.ffill().bfill()
    return trend


def safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(dt) or dt <= 0:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, dt)