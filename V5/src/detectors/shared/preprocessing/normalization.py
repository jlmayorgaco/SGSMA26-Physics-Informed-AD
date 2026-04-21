from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class NormalizationState:
    feature_columns: list[str]
    medians: dict[str, float]
    scales: dict[str, float]


def fit_normalization_state(frame: pd.DataFrame, feature_columns: list[str], clip_quantile: float = 0.995) -> NormalizationState:
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for col in feature_columns:
        series = pd.to_numeric(frame[col], errors="coerce").astype(float).fillna(0.0)
        med = float(series.median())
        hi = float(series.quantile(clip_quantile))
        lo = float(series.quantile(1.0 - clip_quantile))
        scale = max(abs(hi - lo), 1e-6)
        medians[col] = med
        scales[col] = scale
    return NormalizationState(feature_columns=list(feature_columns), medians=medians, scales=scales)


def apply_normalization(frame: pd.DataFrame, state: NormalizationState) -> pd.DataFrame:
    out = frame.copy()
    for col in state.feature_columns:
        if col not in out.columns:
            out[col] = 0.0
        values = pd.to_numeric(out[col], errors="coerce").astype(float).fillna(0.0)
        med = state.medians.get(col, 0.0)
        scale = state.scales.get(col, 1.0)
        values = values.clip(med - 5.0 * scale, med + 5.0 * scale)
        values = (values - med) / max(scale, 1e-6)
        out[col] = values.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
