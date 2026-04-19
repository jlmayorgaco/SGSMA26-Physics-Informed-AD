"""Core m1 normalization math and per-bus transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.normalization_baselines import robust_center, robust_trimmed_mean


def calculate_angular_speed(angle_deg_series: pd.Series, time_index: pd.Index) -> np.ndarray:
    """Convert angle in degrees to angular speed in rad/s."""
    rad_angles = np.deg2rad(pd.to_numeric(angle_deg_series, errors="coerce").to_numpy(dtype=float))
    time_vals = np.asarray(time_index, dtype=float)

    if len(rad_angles) == 0:
        return np.array([], dtype=float)

    # Fill occasional NaN gaps before unwrap/gradient.
    if np.isnan(rad_angles).any():
        valid = np.where(np.isfinite(rad_angles))[0]
        if len(valid) == 0:
            return np.zeros_like(rad_angles)
        rad_angles = np.interp(np.arange(len(rad_angles)), valid, rad_angles[valid])

    unwrapped_rad = np.unwrap(rad_angles)
    if len(unwrapped_rad) < 2:
        return np.zeros_like(unwrapped_rad)
    return np.gradient(unwrapped_rad, time_vals)


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    return ((values + 180.0) % 360.0) - 180.0


def circular_mean_degrees(values: pd.Series) -> float:
    """Compute circular mean in degrees."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0.0
    rad = np.deg2rad(clean)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def detect_signal_family(column_name: str) -> str:
    """Classify raw signal into legacy m1 normalization family."""
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


def normalize_bus_data(
    bus_data: dict[str, pd.DataFrame],
    event_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Normalize all buses using legacy-compatible m1 behavior."""
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

        for col in df.columns:
            if col in ["DATA_PRESENT", "Event"]:
                continue

            family = detect_signal_family(col)
            series = pd.to_numeric(df[col], errors="coerce")
            normal_series = pd.to_numeric(normal_df[col], errors="coerce") if col in normal_df.columns else series

            if family in ["voltage_angle", "current_angle"]:
                out_col = col.replace("ANG", "ANG_SPEED_RAD_S")
                norm_df[out_col] = calculate_angular_speed(series, df.index)
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": out_col,
                        "signal_family": family,
                        "transform": "angle_to_angular_speed_rad_s",
                        "baseline_method": "none",
                        "baseline_value": np.nan,
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "ANG signals are represented as angular speed in rad/s.",
                    }
                )
            elif family in ["voltage_mag", "current_mag", "frequency"]:
                baseline_value, baseline_method = robust_trimmed_mean(normal_series)
                if abs(baseline_value) < 1e-9:
                    baseline_value = 1.0
                    baseline_method = "fallback_constant"
                norm_df[col] = series / baseline_value
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "divide_by_baseline",
                        "baseline_method": baseline_method,
                        "baseline_value": float(baseline_value),
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "PU normalization with robust normal-state baseline.",
                    }
                )
            elif family == "rocof":
                baseline_value, baseline_method = robust_center(normal_series)
                norm_df[col] = series - baseline_value
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "subtract_center",
                        "baseline_method": baseline_method,
                        "baseline_value": float(baseline_value),
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "ROCOF centered around robust normal-state center.",
                    }
                )
            else:
                norm_df[col] = series
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "pass_through",
                        "baseline_method": "none",
                        "baseline_value": np.nan,
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "Unrecognized signal family, left unchanged.",
                    }
                )

        normalized_data[bus] = norm_df

    baseline_df = pd.DataFrame(baseline_rows)
    return normalized_data, baseline_df
