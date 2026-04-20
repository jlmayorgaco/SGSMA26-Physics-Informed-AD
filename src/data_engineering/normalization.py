"""Core m1 normalization orchestration by signal family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.angle_features import build_angle_feature_block
from src.data_engineering.frequency_features import build_frequency_feature_block
from src.data_engineering.magnitude_features import build_magnitude_feature_block
from src.data_engineering.normalization_baselines import robust_center


def detect_signal_family(column_name: str) -> str:
    """Classify one raw PMU signal by family."""
    col = column_name.upper()
    if col.endswith("_ANG"):
        if any(tag in col for tag in ["_IA_ANG", "_IB_ANG", "_IC_ANG"]):
            return "current_angle"
        return "voltage_angle"
    if col.endswith("_MAG"):
        if any(tag in col for tag in ["_IA_MAG", "_IB_MAG", "_IC_MAG"]):
            return "current_mag"
        return "voltage_mag"
    if col.endswith("_FREQ"):
        return "frequency"
    if col.endswith("_ROCOF"):
        return "rocof"
    return "other"


def _append_rows(
    baseline_rows: list[dict],
    rows: list[dict],
    *,
    bus_id: str,
    raw_signal: str,
    signal_family: str,
    n_normal_samples: int,
    baseline_source: str,
) -> None:
    for row in rows:
        baseline_rows.append(
            {
                "bus_id": bus_id,
                "raw_signal": raw_signal,
                "output_signal": row["output_signal"],
                "signal_family": signal_family,
                "transform": row["transform"],
                "baseline_method": row["baseline_method"],
                "baseline_value": row["baseline_value"],
                "n_normal_samples": n_normal_samples,
                "baseline_source": baseline_source,
                "notes": row["notes"],
            }
        )


def normalize_bus_data(
    bus_data: dict[str, pd.DataFrame],
    event_df: pd.DataFrame,
    feature_mode: str = "legacy_replace_angles",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Normalize bus data with explicit feature-mode control.

    Modes:
    - `legacy_replace_angles`: parity mode matching legacy m1 behavior.
    - `augment_angles`: keep original angles and add derived features.
    """
    if feature_mode not in {"legacy_replace_angles", "augment_angles"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    normalized_data: dict[str, pd.DataFrame] = {}
    baseline_rows: list[dict] = []

    global_state = event_df.max(axis=1)
    normal_timestamps = global_state[global_state == 0].index

    for bus, df in bus_data.items():
        normal_df = df.loc[df.index.intersection(normal_timestamps)]
        if normal_df.empty:
            normal_df = df
            normal_source = "full_series_fallback"
        else:
            normal_source = "global_normal_event0"

        norm_df = pd.DataFrame(index=df.index)
        norm_df["DATA_PRESENT"] = pd.to_numeric(df["DATA_PRESENT"], errors="coerce")
        norm_df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
        raw_signal_cols = [col for col in df.columns if col not in ["DATA_PRESENT", "Event"]]
        if feature_mode == "augment_angles":
            for col in raw_signal_cols:
                norm_df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in df.columns:
            if col in ["DATA_PRESENT", "Event"]:
                continue

            family = detect_signal_family(col)
            series = pd.to_numeric(df[col], errors="coerce")
            normal_series = pd.to_numeric(normal_df[col], errors="coerce") if col in normal_df.columns else series
            n_normal = int(normal_series.dropna().shape[0])

            if family in {"voltage_angle", "current_angle"}:
                block, rows = build_angle_feature_block(
                    series=series,
                    normal_series=normal_series,
                    output_prefix=col,
                    time_index=df.index,
                    feature_mode=feature_mode,
                )
            elif family in {"voltage_mag", "current_mag"}:
                block, rows = build_magnitude_feature_block(
                    series=series,
                    normal_series=normal_series,
                    output_name=col,
                    signal_family=family,
                    feature_mode=feature_mode,
                )
            elif family == "frequency":
                block, rows = build_frequency_feature_block(
                    series=series,
                    normal_series=normal_series,
                    output_name=col,
                    feature_mode=feature_mode,
                )
            elif family == "rocof":
                baseline_value, baseline_method = robust_center(normal_series)
                block = pd.DataFrame(index=df.index)
                if feature_mode == "legacy_replace_angles":
                    block[col] = series - baseline_value
                    rows = [
                        {
                            "output_signal": col,
                            "transform": "subtract_center",
                            "baseline_method": baseline_method,
                            "baseline_value": float(baseline_value),
                            "notes": "ROCOF centered around robust normal-state center.",
                        }
                    ]
                else:
                    centered_col = f"{col}_CENTERED"
                    block[col] = series
                    block[centered_col] = series - baseline_value
                    rows = [
                        {
                            "output_signal": col,
                            "transform": "preserve_raw",
                            "baseline_method": "none",
                            "baseline_value": np.nan,
                            "notes": "Original ROCOF signal preserved in augment mode.",
                        },
                        {
                            "output_signal": centered_col,
                            "transform": "subtract_center",
                            "baseline_method": baseline_method,
                            "baseline_value": float(baseline_value),
                            "notes": "ROCOF centered around robust normal-state center.",
                        },
                    ]
            else:
                block = pd.DataFrame(index=df.index)
                block[col] = series
                rows = [
                    {
                        "output_signal": col,
                        "transform": "pass_through",
                        "baseline_method": "none",
                        "baseline_value": np.nan,
                        "notes": "Unrecognized signal family, left unchanged.",
                    }
                ]

            if feature_mode == "augment_angles":
                for new_col in block.columns:
                    if new_col in raw_signal_cols:
                        continue
                    norm_df[new_col] = block[new_col]
            else:
                for new_col in block.columns:
                    norm_df[new_col] = block[new_col]

            _append_rows(
                baseline_rows=baseline_rows,
                rows=rows,
                bus_id=bus,
                raw_signal=col,
                signal_family=family,
                n_normal_samples=n_normal,
                baseline_source=normal_source,
            )

        normalized_data[bus] = norm_df

    baseline_df = pd.DataFrame(baseline_rows)
    return normalized_data, baseline_df
