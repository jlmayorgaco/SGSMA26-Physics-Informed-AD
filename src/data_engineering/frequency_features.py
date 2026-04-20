"""Frequency-domain feature helpers for m1 normalization."""

from __future__ import annotations

import re

import pandas as pd

from src.data_engineering.normalization_baselines import robust_trimmed_mean


def _derive_freq_dev_name(output_name: str) -> str:
    if re.search(r"_freq$", output_name, flags=re.IGNORECASE):
        return re.sub(r"_freq$", "_Freq_DEV_HZ", output_name, flags=re.IGNORECASE)
    return f"{output_name}_DEV_HZ"


def build_frequency_feature_block(
    series: pd.Series,
    normal_series: pd.Series,
    output_name: str,
    feature_mode: str = "legacy_replace_angles",
) -> tuple[pd.DataFrame, list[dict]]:
    """Build normalized frequency features with optional augment derivatives."""
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
        pu_col = re.sub(r"_freq$", "_Freq_PU", output_name, flags=re.IGNORECASE)
        dev_col = _derive_freq_dev_name(output_name)
        block[output_name] = raw_vals
        block[pu_col] = raw_vals / baseline_value
        block[dev_col] = raw_vals - baseline_value
        rows.extend(
            [
                {
                    "output_signal": output_name,
                    "transform": "preserve_raw",
                    "baseline_method": "none",
                    "baseline_value": float("nan"),
                    "notes": "Original frequency signal preserved in augment mode.",
                },
                {
                    "output_signal": pu_col,
                    "transform": "divide_by_baseline",
                    "baseline_method": baseline_method,
                    "baseline_value": float(baseline_value),
                    "notes": "Frequency normalized to per-unit.",
                },
                {
                    "output_signal": dev_col,
                    "transform": "subtract_baseline",
                    "baseline_method": baseline_method,
                    "baseline_value": float(baseline_value),
                    "notes": "Absolute frequency deviation from baseline in Hz.",
                },
            ]
        )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    return block, rows
