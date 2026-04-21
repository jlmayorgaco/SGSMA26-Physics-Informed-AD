from __future__ import annotations

import numpy as np
import pandas as pd


def wrap_degrees(angle_deg: np.ndarray | pd.Series) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return ((angle + 180.0) % 360.0) - 180.0


def add_angle_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    angle_cols = [c for c in out.columns if c.endswith("_ANG")]
    for col in angle_cols:
        angle = wrap_degrees(pd.to_numeric(out[col], errors="coerce").fillna(0.0))
        rad = np.deg2rad(angle)
        out[f"{col}__sin"] = np.sin(rad)
        out[f"{col}__cos"] = np.cos(rad)
        out[f"{col}__wrapped"] = angle
    return out

