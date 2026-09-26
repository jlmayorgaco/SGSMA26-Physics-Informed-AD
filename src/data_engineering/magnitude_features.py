"""Magnitude-domain feature helpers for m1 normalization."""

from __future__ import annotations

import pandas as pd

from src.data_engineering.normalization_baselines import robust_trimmed_mean


def build_magnitude_feature_block(
    series: pd.Series,
    normal_series: pd.Series,
    output_name: str,
    signal_family: str,
    feature_mode: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build normalized magnitude features for one signal."""
    baseline_value, baseline_method = robust_trimmed_mean(normal_series)
    if abs(baseline_value) < 1e-9:
        baseline_value = 1.0
        baseline_method = "fallback_constant"

    raw_vals = pd.to_numeric(series, errors="coerce")
    block = pd.DataFrame(index=series.index)
    rows: list[dict] = []

    if feature_mode == "legacy_replace_angles":
        block[output_name] = raw_vals / baseline_value
        rows.append(
            {
                "output_signal": output_name,
                "transform": "divide_by_baseline",
                "baseline_method": baseline_method,
                "baseline_value": float(baseline_value),
                "notes": "PU normalization with robust normal-state baseline.",
            }
        )
    elif feature_mode == "augment_angles":
        pu_col = output_name.replace("_MAG", "_MAG_PU")
        dev_col = output_name.replace("_MAG", "_MAG_DEV_PU")
        pu_vals = raw_vals / baseline_value
        block[output_name] = raw_vals
        block[pu_col] = pu_vals
        block[dev_col] = pu_vals - 1.0
        rows.extend(
            [
                {
                    "output_signal": output_name,
                    "transform": "preserve_raw",
                    "baseline_method": "none",
                    "baseline_value": float("nan"),
                    "notes": "Original magnitude signal preserved in augment mode.",
                },
                {
                    "output_signal": pu_col,
                    "transform": "divide_by_baseline",
                    "baseline_method": baseline_method,
                    "baseline_value": float(baseline_value),
                    "notes": "Magnitude normalized to per-unit.",
                },
                {
                    "output_signal": dev_col,
                    "transform": "pu_minus_1",
                    "baseline_method": "constant",
                    "baseline_value": 1.0,
                    "notes": "Deviation in per-unit from normalized unity baseline.",
                },
            ]
        )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    _ = signal_family
    return block, rows
