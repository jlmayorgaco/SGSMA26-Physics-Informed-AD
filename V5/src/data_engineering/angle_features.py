"""Angle-domain feature helpers for m1 normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_angular_speed(angle_deg_series: pd.Series, time_index: pd.Index) -> np.ndarray:
    """Convert angle (deg) into angular speed (rad/s) with legacy-compatible behavior."""
    rad_angles = np.deg2rad(pd.to_numeric(angle_deg_series, errors="coerce").to_numpy(dtype=float))
    time_vals = np.asarray(time_index, dtype=float)

    if len(rad_angles) == 0:
        return np.array([], dtype=float)

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
    """Wrap angles to [-180, 180)."""
    return ((values + 180.0) % 360.0) - 180.0


def circular_mean_degrees(values: pd.Series) -> float:
    """Compute circular mean in degrees."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0.0
    rad = np.deg2rad(clean)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def build_angle_feature_block(
    series: pd.Series,
    normal_series: pd.Series,
    output_prefix: str,
    time_index: pd.Index,
    feature_mode: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build angle-derived feature columns for one raw angular signal."""
    block = pd.DataFrame(index=time_index)
    rows: list[dict] = []

    speed_col = output_prefix.replace("ANG", "ANG_SPEED_RAD_S")
    speed_vals = calculate_angular_speed(series, time_index)

    if feature_mode == "legacy_replace_angles":
        block[speed_col] = speed_vals
        rows.append(
            {
                "output_signal": speed_col,
                "transform": "angle_to_angular_speed_rad_s",
                "baseline_method": "none",
                "baseline_value": np.nan,
                "notes": "ANG signals are represented as angular speed in rad/s.",
            }
        )
        return block, rows

    if feature_mode != "augment_angles":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    raw_vals = pd.to_numeric(series, errors="coerce")
    block[output_prefix] = raw_vals
    block[speed_col] = speed_vals

    sin_col = output_prefix.replace("ANG", "ANG_SIN")
    cos_col = output_prefix.replace("ANG", "ANG_COS")
    block[sin_col] = np.sin(np.deg2rad(raw_vals.to_numpy(dtype=float)))
    block[cos_col] = np.cos(np.deg2rad(raw_vals.to_numpy(dtype=float)))

    baseline_angle = circular_mean_degrees(normal_series)
    dev_col = output_prefix.replace("ANG", "ANG_DEV_DEG")
    block[dev_col] = wrap_degrees(raw_vals.to_numpy(dtype=float) - baseline_angle)

    rows.extend(
        [
            {
                "output_signal": output_prefix,
                "transform": "preserve_raw",
                "baseline_method": "none",
                "baseline_value": np.nan,
                "notes": "Original angular signal preserved in augment mode.",
            },
            {
                "output_signal": speed_col,
                "transform": "angle_to_angular_speed_rad_s",
                "baseline_method": "none",
                "baseline_value": np.nan,
                "notes": "Derived angular speed in rad/s.",
            },
            {
                "output_signal": sin_col,
                "transform": "sin_deg",
                "baseline_method": "none",
                "baseline_value": np.nan,
                "notes": "Circular embedding sine component.",
            },
            {
                "output_signal": cos_col,
                "transform": "cos_deg",
                "baseline_method": "none",
                "baseline_value": np.nan,
                "notes": "Circular embedding cosine component.",
            },
            {
                "output_signal": dev_col,
                "transform": "wrap_relative_to_circular_mean",
                "baseline_method": "circular_mean",
                "baseline_value": float(baseline_angle),
                "notes": "Wrapped angular deviation relative to circular baseline.",
            },
        ]
    )
    return block, rows
