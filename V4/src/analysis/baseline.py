from __future__ import annotations

import pandas as pd

from src.config.constants import MEASUREMENT_COLUMNS


def filter_normal_operation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only normal operation frames:
    - Event == 0
    - DATA_PRESENT == 1
    """
    out = df.copy()

    if "Event" in out.columns:
        out = out[out["Event"].fillna(-1).astype(int) == 0]

    if "DATA_PRESENT" in out.columns:
        out = out[out["DATA_PRESENT"].fillna(0).astype(float) > 0.5]

    return out.reset_index(drop=True)


def filter_finite_signal(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out = out[out[column].notna()]
    return out.reset_index(drop=True)


def filter_normal_operation_for_signal(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = filter_normal_operation(df)
    return filter_finite_signal(out, column)


def build_normal_operation_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if "Event" in df.columns:
        mask &= df["Event"].fillna(-1).astype(int) == 0

    if "DATA_PRESENT" in df.columns:
        mask &= df["DATA_PRESENT"].fillna(0).astype(float) > 0.5

    return mask


def normal_operation_summary(df: pd.DataFrame) -> dict:
    mask = build_normal_operation_mask(df)
    total = len(df)
    kept = int(mask.sum())

    return {
        "total_rows": int(total),
        "normal_rows": kept,
        "normal_ratio": float(kept / total) if total else 0.0,
        "usable_measurement_columns": [c for c in MEASUREMENT_COLUMNS if c in df.columns],
    }