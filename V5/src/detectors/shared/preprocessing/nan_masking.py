from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_feature_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    blocked = exclude or set()
    out: list[str] = []
    for col in frame.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out.append(col)
    return sorted(out)


def apply_nan_masking(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    fill_method: str = "ffill_bfill",
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    cols = feature_columns or numeric_feature_columns(
        out, exclude={"EVENT", "DATA_PRESENT", "EVENT_COARSE", "SCENARIO_ID", "SPLIT"}
    )
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = out[cols].isna().astype(int).rename(columns={c: f"{c}__is_nan" for c in cols})
    if fill_method == "ffill_bfill":
        out[cols] = out[cols].ffill().bfill()
    elif fill_method == "bfill_only":
        out[cols] = out[cols].bfill()
    elif fill_method == "ffill_limit1":
        out[cols] = out[cols].ffill(limit=1).bfill(limit=1)
    elif fill_method == "interpolate":
        out[cols] = out[cols].interpolate(limit_direction="both")
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return out, mask
